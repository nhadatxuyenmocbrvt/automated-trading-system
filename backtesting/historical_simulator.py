#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module mô phỏng lịch sử thị trường.

Module này cung cấp các lớp và phương thức để mô phỏng dữ liệu lịch sử thị trường
và tái tạo lại các điều kiện giao dịch trong quá khứ cho mục đích backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
import random
from copy import deepcopy
import json

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import Timeframe, OrderType, OrderStatus, PositionSide
from environments.base_environment import BaseEnvironment
from environments.trading_gym.trading_env import TradingEnv

class HistoricalSimulator:
    """
    Lớp mô phỏng dữ liệu lịch sử thị trường.
    
    Lớp này mô phỏng các điều kiện thị trường trong quá khứ dựa trên dữ liệu lịch sử thực tế,
    cung cấp một môi trường giao dịch mô phỏng cho việc backtesting các chiến lược.
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame] = None,
        data_path: Optional[str] = None,
        symbols: List[str] = None,
        timeframe: str = "1h",
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001,
        slippage: float = 0.0,
        liquidity_constraints: bool = False,
        spread: Optional[float] = None,
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        trading_hours: Optional[Dict[str, Tuple[str, str]]] = None,
        trading_days: Optional[List[int]] = None,
        holidays: Optional[List[Union[str, datetime]]] = None,
        min_order_size: Dict[str, float] = None,
        enable_fractional: bool = True,
        max_leverage: float = 1.0,
        apply_funding_rate: bool = False,
        funding_rate: Dict[str, float] = None,
        funding_interval: str = "8h",
        random_events: bool = False,
        event_probability: float = 0.05,
        latency_model: Optional[Callable] = None,
        cache_data: bool = True
    ):
        """
        Khởi tạo lớp HistoricalSimulator.
        
        Args:
            data (Dict[str, pd.DataFrame], optional): Dữ liệu lịch sử cho mỗi symbol.
            data_path (str, optional): Đường dẫn đến dữ liệu lịch sử.
            symbols (List[str], optional): Danh sách các symbol cần mô phỏng.
            timeframe (str, optional): Khung thời gian mô phỏng (e.g., "1m", "1h", "1d").
            start_date (Union[str, datetime], optional): Ngày bắt đầu mô phỏng.
            end_date (Union[str, datetime], optional): Ngày kết thúc mô phỏng.
            initial_balance (float, optional): Số dư ban đầu. Mặc định là 10000.0.
            fee_rate (float, optional): Tỷ lệ phí giao dịch. Mặc định là 0.001 (0.1%).
            slippage (float, optional): Tỷ lệ trượt giá. Mặc định là 0.0 (0%).
            liquidity_constraints (bool, optional): Bật/tắt ràng buộc thanh khoản. Mặc định là False.
            spread (float, optional): Spread giữa giá mua và bán. Mặc định là None.
            seed (int, optional): Seed cho random generator. Mặc định là None.
            logger (logging.Logger, optional): Logger tùy chỉnh.
            trading_hours (Dict[str, Tuple[str, str]], optional): Giờ giao dịch cho mỗi symbol, dạng {"symbol": ("HH:MM", "HH:MM")}.
            trading_days (List[int], optional): Các ngày trong tuần có thể giao dịch (0=Thứ Hai, 6=Chủ Nhật).
            holidays (List[Union[str, datetime]], optional): Danh sách các ngày lễ không giao dịch.
            min_order_size (Dict[str, float], optional): Kích thước đặt lệnh tối thiểu cho mỗi symbol.
            enable_fractional (bool, optional): Cho phép giao dịch số lẻ. Mặc định là True.
            max_leverage (float, optional): Đòn bẩy tối đa. Mặc định là 1.0 (spot).
            apply_funding_rate (bool, optional): Áp dụng phí funding rate cho vị thế. Mặc định là False.
            funding_rate (Dict[str, float], optional): Funding rate cho mỗi symbol.
            funding_interval (str, optional): Khoảng thời gian tính funding rate. Mặc định là "8h".
            random_events (bool, optional): Bật/tắt sự kiện ngẫu nhiên. Mặc định là False.
            event_probability (float, optional): Xác suất xảy ra sự kiện ngẫu nhiên. Mặc định là 0.05.
            latency_model (Callable, optional): Hàm mô phỏng độ trễ khi đặt lệnh.
            cache_data (bool, optional): Bật/tắt cache dữ liệu. Mặc định là True.
        """
        self.logger = logger or get_logger("historical_simulator")
        
        # Thiết lập seed nếu được cung cấp
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Dữ liệu
        self.data = data or {}
        self.data_path = data_path
        self.symbols = symbols or list(self.data.keys())
        self.timeframe = timeframe
        
        # Tải dữ liệu nếu có đường dẫn
        if data_path and not self.data:
            self._load_data()
        
        # Thời gian
        self.start_date = self._parse_date(start_date) if start_date else None
        self.end_date = self._parse_date(end_date) if end_date else None
        
        # Tài khoản
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.max_leverage = max_leverage
        
        # Ràng buộc thị trường
        self.liquidity_constraints = liquidity_constraints
        self.spread = spread
        self.min_order_size = min_order_size or {}
        self.enable_fractional = enable_fractional
        
        # Lịch giao dịch
        self.trading_hours = trading_hours or {}
        self.trading_days = trading_days or list(range(7))  # Mặc định là tất cả các ngày
        self.holidays = [self._parse_date(date) for date in holidays] if holidays else []
        
        # Funding rate (cho futures)
        self.apply_funding_rate = apply_funding_rate
        self.funding_rate = funding_rate or {}
        self.funding_interval = funding_interval
        
        # Sự kiện ngẫu nhiên
        self.random_events = random_events
        self.event_probability = event_probability
        
        # Độ trễ
        self.latency_model = latency_model or (lambda: 0)  # Mặc định không có độ trễ
        
        # Cache
        self.cache_data = cache_data
        self._cached_slices = {}
        
        # Trạng thái mô phỏng
        self.current_timestamp = None
        self.current_timestep = 0
        self.current_prices = {}
        self.current_orderbooks = {}
        self.open_positions = {}
        self.pending_orders = []
        self.execution_history = []
        self.balance_history = []
        self.portfolio_history = []
        
        # Môi trường mô phỏng
        self.environments = {}
        
        # Chuẩn bị dữ liệu
        self._prepare_data()
        
        self.logger.info(f"Đã khởi tạo HistoricalSimulator với {len(self.symbols)} symbols từ {self.start_date} đến {self.end_date}")
    
    def _parse_date(self, date: Union[str, datetime]) -> datetime:
        """
        Chuyển đổi ngày từ dạng chuỗi sang datetime.
        
        Args:
            date (Union[str, datetime]): Ngày dưới dạng chuỗi hoặc datetime.
            
        Returns:
            datetime: Đối tượng datetime.
        """
        if isinstance(date, datetime):
            return date
        
        # Thử một số định dạng phổ biến
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d",
            "%d-%m-%Y %H:%M:%S",
            "%d-%m-%Y %H:%M",
            "%d-%m-%Y",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date, fmt)
            except ValueError:
                continue
        
        # Nếu không có định dạng nào khớp
        raise ValueError(f"Không thể chuyển đổi ngày '{date}' sang định dạng datetime")
    
    def _load_data(self) -> None:
        """
        Tải dữ liệu từ đường dẫn đã chỉ định.
        """
        import os
        import glob
        
        # Kiểm tra đường dẫn
        if not os.path.exists(self.data_path):
            self.logger.error(f"Đường dẫn dữ liệu không tồn tại: {self.data_path}")
            return
        
        # Tìm tất cả các file CSV/Parquet trong thư mục
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        parquet_files = glob.glob(os.path.join(self.data_path, "*.parquet"))
        
        all_files = csv_files + parquet_files
        
        if not all_files:
            self.logger.warning(f"Không tìm thấy file dữ liệu trong: {self.data_path}")
            return
        
        # Tải các file
        for file_path in all_files:
            try:
                file_name = os.path.basename(file_path)
                symbol = os.path.splitext(file_name)[0].upper()
                
                # Đọc file dựa vào định dạng
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                else:  # .parquet
                    df = pd.read_parquet(file_path)
                
                # Chuyển đổi cột timestamp sang datetime nếu có
                if "timestamp" in df.columns:
                    if pd.api.types.is_string_dtype(df["timestamp"]):
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                    
                    # Đặt làm index
                    df.set_index("timestamp", inplace=True)
                
                # Lưu vào dict dữ liệu
                self.data[symbol] = df
                
                # Thêm vào danh sách symbols nếu chưa có
                if symbol not in self.symbols:
                    self.symbols.append(symbol)
                
                self.logger.info(f"Đã tải {len(df)} dòng dữ liệu cho {symbol} từ {file_path}")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi tải file {file_path}: {str(e)}")
    
    def _prepare_data(self) -> None:
        """
        Chuẩn bị dữ liệu cho mô phỏng.
        """
        if not self.data:
            self.logger.warning("Không có dữ liệu để chuẩn bị")
            return
        
        # Xác định thời gian bắt đầu và kết thúc
        if self.start_date is None or self.end_date is None:
            all_dates = []
            
            for symbol, df in self.data.items():
                if not df.empty:
                    all_dates.extend(df.index)
            
            if all_dates:
                if self.start_date is None:
                    self.start_date = min(all_dates)
                    
                if self.end_date is None:
                    self.end_date = max(all_dates)
        
        # Lọc dữ liệu theo khoảng thời gian
        for symbol, df in self.data.items():
            try:
                # Kiểm tra xem có phải DataFrame không
                if not isinstance(df, pd.DataFrame):
                    self.logger.error(f"Dữ liệu cho {symbol} không phải là DataFrame")
                    continue
                
                # Kiểm tra rỗng
                if df.empty:
                    self.logger.warning(f"DataFrame rỗng cho {symbol}")
                    continue
                
                # Đảm bảo index là datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    self.logger.warning(f"Index của {symbol} không phải là DatetimeIndex, thử chuyển đổi")
                    try:
                        if "timestamp" in df.columns:
                            df.set_index("timestamp", inplace=True)
                        elif "date" in df.columns:
                            df.set_index("date", inplace=True)
                        else:
                            raise ValueError("Không tìm thấy cột thời gian phù hợp")
                    except Exception as e:
                        self.logger.error(f"Không thể chuyển đổi index của {symbol}: {str(e)}")
                        continue
                
                # Lọc theo khoảng thời gian
                if self.start_date is not None:
                    df = df[df.index >= self.start_date]
                
                if self.end_date is not None:
                    df = df[df.index <= self.end_date]
                
                # Đảm bảo có các cột cần thiết
                required_cols = ["open", "high", "low", "close"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    self.logger.warning(f"Thiếu các cột {missing_cols} cho {symbol}")
                
                # Cập nhật lại DataFrame đã lọc
                self.data[symbol] = df
                
                # Kiểm tra lại sau khi lọc
                if df.empty:
                    self.logger.warning(f"DataFrame rỗng cho {symbol} sau khi lọc theo thời gian")
                    continue
                
                # Kiểm tra timestamp đầu tiên và cuối cùng
                self.logger.info(f"Dữ liệu {symbol}: từ {df.index[0]} đến {df.index[-1]}, {len(df)} dòng")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi chuẩn bị dữ liệu cho {symbol}: {str(e)}")
    
    def _get_data_slice(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Lấy phần dữ liệu trong khoảng thời gian chỉ định cho symbol.
        
        Args:
            symbol (str): Symbol cần lấy dữ liệu.
            start_time (datetime): Thời gian bắt đầu.
            end_time (datetime): Thời gian kết thúc.
            
        Returns:
            pd.DataFrame: DataFrame chứa dữ liệu trong khoảng thời gian.
        """
        # Kiểm tra cache
        cache_key = f"{symbol}_{start_time}_{end_time}"
        if self.cache_data and cache_key in self._cached_slices:
            return self._cached_slices[cache_key]
        
        # Lấy dữ liệu nếu symbol tồn tại
        if symbol not in self.data:
            self.logger.warning(f"Symbol {symbol} không có trong dữ liệu")
            return pd.DataFrame()
        
        df = self.data[symbol]
        
        # Lọc theo khoảng thời gian
        slice_df = df[(df.index >= start_time) & (df.index <= end_time)]
        
        # Lưu vào cache nếu cần
        if self.cache_data:
            self._cached_slices[cache_key] = slice_df
        
        return slice_df
    
    def _is_trading_time(self, timestamp: datetime, symbol: str) -> bool:
        """
        Kiểm tra xem timestamp có nằm trong thời gian giao dịch không.
        
        Args:
            timestamp (datetime): Thời gian cần kiểm tra.
            symbol (str): Symbol cần kiểm tra.
            
        Returns:
            bool: True nếu là thời gian giao dịch, False nếu không.
        """
        # Kiểm tra ngày nghỉ
        if timestamp.date() in [h.date() for h in self.holidays]:
            return False
        
        # Kiểm tra ngày trong tuần
        if timestamp.weekday() not in self.trading_days:
            return False
        
        # Kiểm tra giờ giao dịch nếu có
        if symbol in self.trading_hours:
            start_hour, end_hour = self.trading_hours[symbol]
            
            # Chuyển đổi sang datetime
            start_time = datetime.strptime(start_hour, "%H:%M").time()
            end_time = datetime.strptime(end_hour, "%H:%M").time()
            
            current_time = timestamp.time()
            
            # Kiểm tra trong khoảng thời gian giao dịch
            if start_time <= end_time:
                # Trường hợp thông thường (ví dụ: 9:00 - 17:00)
                return start_time <= current_time <= end_time
            else:
                # Trường hợp qua ngày (ví dụ: 22:00 - 04:00)
                return current_time >= start_time or current_time <= end_time
        
        # Mặc định là luôn trong thời gian giao dịch
        return True
    
    def _apply_slippage(self, price: float, side: str) -> float:
        """
        Áp dụng slippage vào giá.
        
        Args:
            price (float): Giá gốc.
            side (str): Phía giao dịch ("buy" hoặc "sell").
            
        Returns:
            float: Giá sau khi áp dụng slippage.
        """
        if self.slippage == 0:
            return price
        
        # Slippage tính theo phần trăm
        slippage_amount = price * self.slippage
        
        # Áp dụng slippage vào giá
        if side.lower() == "buy":
            return price * (1 + self.slippage)
        else:  # sell
            return price * (1 - self.slippage)
    
    def _apply_spread(self, price: float, side: str) -> float:
        """
        Áp dụng spread vào giá.
        
        Args:
            price (float): Giá gốc.
            side (str): Phía giao dịch ("buy" hoặc "sell").
            
        Returns:
            float: Giá sau khi áp dụng spread.
        """
        if self.spread is None or self.spread == 0:
            return price
        
        # Spread tính theo phần trăm
        spread_amount = price * (self.spread / 2)
        
        # Áp dụng spread vào giá
        if side.lower() == "buy":
            return price + spread_amount
        else:  # sell
            return price - spread_amount
    
    def _check_liquidity(self, symbol: str, price: float, size: float, side: str) -> Tuple[bool, float]:
        """
        Kiểm tra và điều chỉnh lệnh theo thanh khoản.
        
        Args:
            symbol (str): Symbol giao dịch.
            price (float): Giá đặt lệnh.
            size (float): Số lượng ban đầu.
            side (str): Phía giao dịch ("buy" hoặc "sell").
            
        Returns:
            Tuple[bool, float]: (có đủ thanh khoản, số lượng điều chỉnh).
        """
        if not self.liquidity_constraints:
            return True, size
        
        if symbol not in self.current_orderbooks:
            return True, size
        
        orderbook = self.current_orderbooks[symbol]
        
        if side.lower() == "buy":
            if "asks" not in orderbook or not orderbook["asks"]:
                return True, size
                
            # Tính tổng lượng có thể mua
            available_size = sum(level[1] for level in orderbook["asks"] if level[0] <= price)
            
            if available_size >= size:
                return True, size
            else:
                return False, available_size
        else:  # sell
            if "bids" not in orderbook or not orderbook["bids"]:
                return True, size
                
            # Tính tổng lượng có thể bán
            available_size = sum(level[1] for level in orderbook["bids"] if level[0] >= price)
            
            if available_size >= size:
                return True, size
            else:
                return False, available_size
    
    def _execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực thi lệnh giao dịch.
        
        Args:
            order (Dict[str, Any]): Thông tin lệnh giao dịch.
            
        Returns:
            Dict[str, Any]: Kết quả thực thi lệnh.
        """
        symbol = order["symbol"]
        order_type = order["type"]
        side = order["side"]
        size = order["size"]
        
        # Kiểm tra kích thước tối thiểu
        if symbol in self.min_order_size and size < self.min_order_size[symbol]:
            return {
                "success": False,
                "order_id": order["order_id"],
                "reason": f"Kích thước lệnh nhỏ hơn mức tối thiểu ({size} < {self.min_order_size[symbol]})"
            }
        
        # Kiểm tra thời gian giao dịch
        if not self._is_trading_time(self.current_timestamp, symbol):
            return {
                "success": False,
                "order_id": order["order_id"],
                "reason": f"Ngoài thời gian giao dịch cho {symbol}"
            }
        
        # Kiểm tra giá hiện tại
        if symbol not in self.current_prices:
            return {
                "success": False,
                "order_id": order["order_id"],
                "reason": f"Không có giá hiện tại cho {symbol}"
            }
        
        # Xử lý các loại lệnh
        execution_price = None
        filled_size = size
        
        if order_type == OrderType.MARKET.value:
            # Lấy giá hiện tại
            base_price = self.current_prices[symbol]["close"]
            
            # Áp dụng spread và slippage
            execution_price = self._apply_spread(base_price, side)
            execution_price = self._apply_slippage(execution_price, side)
            
            # Kiểm tra thanh khoản
            has_liquidity, adjusted_size = self._check_liquidity(symbol, execution_price, size, side)
            filled_size = adjusted_size if has_liquidity else filled_size
            
            if not has_liquidity and adjusted_size < size:
                self.logger.warning(f"Lệnh market cho {symbol} bị giới hạn bởi thanh khoản: {adjusted_size}/{size}")
            
        elif order_type == OrderType.LIMIT.value:
            # Kiểm tra giá
            limit_price = order["price"]
            current_price = self.current_prices[symbol]["close"]
            
            # Kiểm tra xem lệnh có thể được thực thi ngay không
            if (side == "buy" and limit_price >= current_price) or (side == "sell" and limit_price <= current_price):
                # Đối với lệnh limit có thể thực thi ngay, sử dụng giá limit
                execution_price = limit_price
                
                # Kiểm tra thanh khoản
                has_liquidity, adjusted_size = self._check_liquidity(symbol, execution_price, size, side)
                filled_size = adjusted_size if has_liquidity else filled_size
                
                if not has_liquidity and adjusted_size < size:
                    self.logger.warning(f"Lệnh limit cho {symbol} bị giới hạn bởi thanh khoản: {adjusted_size}/{size}")
            else:
                # Lệnh không thể thực thi ngay
                return {
                    "success": False,
                    "order_id": order["order_id"],
                    "reason": f"Lệnh limit không thể thực thi với giá {limit_price} (hiện tại: {current_price})"
                }
        else:
            # Loại lệnh không được hỗ trợ
            return {
                "success": False,
                "order_id": order["order_id"],
                "reason": f"Loại lệnh không được hỗ trợ: {order_type}"
            }
        
        # Kiểm tra số dư
        total_cost = execution_price * filled_size
        fee = total_cost * self.fee_rate
        
        if side == "buy":
            if total_cost + fee > self.current_balance:
                return {
                    "success": False,
                    "order_id": order["order_id"],
                    "reason": f"Không đủ số dư ({self.current_balance} < {total_cost + fee})"
                }
        
        # Cập nhật số dư
        if side == "buy":
            self.current_balance -= (total_cost + fee)
            
            # Cập nhật vị thế
            if symbol not in self.open_positions:
                self.open_positions[symbol] = {
                    "symbol": symbol,
                    "size": filled_size,
                    "entry_price": execution_price,
                    "side": PositionSide.LONG.value,
                    "entry_time": self.current_timestamp,
                    "unrealized_pnl": 0.0
                }
            else:
                # Cập nhật vị thế hiện tại
                position = self.open_positions[symbol]
                
                if position["side"] == PositionSide.LONG.value:
                    # Tính giá vào mới dựa trên trung bình có trọng số
                    new_size = position["size"] + filled_size
                    new_entry_price = (position["entry_price"] * position["size"] + execution_price * filled_size) / new_size
                    
                    position["size"] = new_size
                    position["entry_price"] = new_entry_price
                else:
                    # Nếu vị thế hiện tại là short, giảm kích thước hoặc đóng
                    if filled_size <= position["size"]:
                        # Tính P&L
                        pnl = (position["entry_price"] - execution_price) * filled_size
                        
                        # Cập nhật vị thế
                        position["size"] -= filled_size
                        
                        # Nếu vị thế đã đóng hoàn toàn
                        if position["size"] == 0:
                            del self.open_positions[symbol]
                        
                        # Cập nhật số dư với P&L
                        self.current_balance += pnl
                    else:
                        # Nếu kích thước lớn hơn, đóng vị thế short và mở vị thế long mới
                        
                        # Tính P&L cho phần đóng
                        pnl = (position["entry_price"] - execution_price) * position["size"]
                        
                        # Phần còn lại để mở vị thế mới
                        remaining_size = filled_size - position["size"]
                        
                        # Cập nhật số dư với P&L
                        self.current_balance += pnl
                        
                        # Tạo vị thế long mới
                        self.open_positions[symbol] = {
                            "symbol": symbol,
                            "size": remaining_size,
                            "entry_price": execution_price,
                            "side": PositionSide.LONG.value,
                            "entry_time": self.current_timestamp,
                            "unrealized_pnl": 0.0
                        }
        else:  # sell
            # Kiểm tra có vị thế không
            if symbol not in self.open_positions or self.open_positions[symbol]["side"] != PositionSide.LONG.value:
                # Mở vị thế short mới
                if self.max_leverage > 1.0:  # Chỉ cho phép short nếu có đòn bẩy
                    self.open_positions[symbol] = {
                        "symbol": symbol,
                        "size": filled_size,
                        "entry_price": execution_price,
                        "side": PositionSide.SHORT.value,
                        "entry_time": self.current_timestamp,
                        "unrealized_pnl": 0.0
                    }
                else:
                    return {
                        "success": False,
                        "order_id": order["order_id"],
                        "reason": f"Không thể short {symbol} vì không có đòn bẩy"
                    }
            else:
                # Đóng một phần hoặc toàn bộ vị thế long
                position = self.open_positions[symbol]
                
                if filled_size <= position["size"]:
                    # Tính P&L
                    pnl = (execution_price - position["entry_price"]) * filled_size
                    
                    # Cập nhật vị thế
                    position["size"] -= filled_size
                    
                    # Nếu vị thế đã đóng hoàn toàn
                    if position["size"] == 0:
                        del self.open_positions[symbol]
                    
                    # Cập nhật số dư với P&L và phí
                    self.current_balance += (total_cost - fee + pnl)
                else:
                    # Không thể bán nhiều hơn vị thế hiện có
                    return {
                        "success": False,
                        "order_id": order["order_id"],
                        "reason": f"Không thể bán {filled_size} vì chỉ có {position['size']} trong vị thế"
                    }
        
        # Tạo kết quả thực thi
        execution_result = {
            "success": True,
            "order_id": order["order_id"],
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "requested_size": size,
            "filled_size": filled_size,
            "execution_price": execution_price,
            "fee": fee,
            "timestamp": self.current_timestamp,
            "balance_after": self.current_balance
        }
        
        # Thêm vào lịch sử thực thi
        self.execution_history.append(execution_result)
        
        # Ghi log
        self.logger.info(f"Đã thực thi lệnh {side} {filled_size} {symbol} @ {execution_price}")
        
        return execution_result
    
    def _process_pending_orders(self) -> None:
        """
        Xử lý các lệnh đang chờ.
        """
        if not self.pending_orders:
            return
        
        # Xử lý từng lệnh
        remain_orders = []
        
        for order in self.pending_orders:
            # Kiểm tra xem lệnh có thể được thực thi với giá hiện tại không
            symbol = order["symbol"]
            
            if symbol not in self.current_prices:
                remain_orders.append(order)
                continue
                
            current_price = self.current_prices[symbol]["close"]
            
            if order["type"] == OrderType.LIMIT.value:
                side = order["side"]
                limit_price = order["price"]
                
                # Kiểm tra điều kiện thực thi
                if (side == "buy" and limit_price >= current_price) or (side == "sell" and limit_price <= current_price):
                    # Thực thi lệnh
                    result = self._execute_order(order)
                    
                    if not result["success"]:
                        # Nếu không thành công, giữ lại trong hàng đợi
                        remain_orders.append(order)
                else:
                    # Nếu chưa đạt điều kiện, giữ lại trong hàng đợi
                    remain_orders.append(order)
            else:
                # Đối với các loại lệnh khác, thực thi ngay
                result = self._execute_order(order)
                
                if not result["success"]:
                    # Nếu không thành công, giữ lại trong hàng đợi
                    remain_orders.append(order)
        
        # Cập nhật lại danh sách lệnh đang chờ
        self.pending_orders = remain_orders
    
    def _calculate_funding_fee(self) -> None:
        """
        Tính phí funding cho các vị thế.
        """
        if not self.apply_funding_rate or not self.open_positions:
            return
        
        # Kiểm tra xem đã đến thời điểm tính funding chưa
        if self.funding_interval == "8h":
            funding_hours = [0, 8, 16]
            if self.current_timestamp.hour not in funding_hours or self.current_timestamp.minute != 0:
                return
        elif self.funding_interval == "1h":
            if self.current_timestamp.minute != 0:
                return
        elif self.funding_interval == "1d":
            if self.current_timestamp.hour != 0 or self.current_timestamp.minute != 0:
                return
        
        # Tính funding fee cho từng vị thế
        for symbol, position in list(self.open_positions.items()):
            if symbol not in self.funding_rate:
                continue
                
            funding_rate = self.funding_rate[symbol]
            position_value = position["size"] * position["entry_price"]
            
            # Tính fee dựa trên phía vị thế
            if position["side"] == PositionSide.LONG.value:
                fee = position_value * funding_rate
            else:  # short
                fee = -position_value * funding_rate
            
            # Trừ phí từ số dư
            self.current_balance -= fee
            
            self.logger.info(f"Đã tính funding fee cho {symbol}: {fee:.4f} (rate: {funding_rate:.6f})")
    
    def _update_positions(self) -> None:
        """
        Cập nhật trạng thái các vị thế mở.
        """
        # Tính unrealized P&L cho các vị thế
        for symbol, position in list(self.open_positions.items()):
            if symbol not in self.current_prices:
                continue
            
            current_price = self.current_prices[symbol]["close"]
            
            if position["side"] == PositionSide.LONG.value:
                position["unrealized_pnl"] = (current_price - position["entry_price"]) * position["size"]
            else:  # short
                position["unrealized_pnl"] = (position["entry_price"] - current_price) * position["size"]
            
            # Mở rộng: kiểm tra stop loss và take profit nếu cần
    
    def _generate_random_events(self) -> None:
        """
        Tạo các sự kiện ngẫu nhiên.
        """
        if not self.random_events:
            return
            
        # Xác suất sự kiện
        if random.random() > self.event_probability:
            return
            
        # Loại sự kiện ngẫu nhiên
        event_types = ["price_spike", "liquidity_dry", "funding_rate_change", "market_halt"]
        event_type = random.choice(event_types)
        
        # Chọn symbol ngẫu nhiên
        affected_symbols = random.sample(self.symbols, min(len(self.symbols), 3))
        
        # Tạo sự kiện
        for symbol in affected_symbols:
            if symbol not in self.current_prices:
                continue
                
            current_price = self.current_prices[symbol]["close"]
            
            if event_type == "price_spike":
                # Tạo đột biến giá
                direction = random.choice([-1, 1])
                spike_pct = random.uniform(0.03, 0.15)  # 3-15%
                
                self.current_prices[symbol]["close"] = current_price * (1 + direction * spike_pct)
                self.logger.info(f"Sự kiện ngẫu nhiên: Đột biến giá {symbol} {direction * spike_pct:.2%}")
                
            elif event_type == "liquidity_dry":
                # Giảm thanh khoản
                if symbol in self.current_orderbooks:
                    for side in ["bids", "asks"]:
                        if side in self.current_orderbooks[symbol]:
                            self.current_orderbooks[symbol][side] = [
                                (level[0], level[1] * random.uniform(0.1, 0.5))
                                for level in self.current_orderbooks[symbol][side]
                            ]
                    
                    self.logger.info(f"Sự kiện ngẫu nhiên: Cạn thanh khoản {symbol}")
                    
            elif event_type == "funding_rate_change":
                # Thay đổi funding rate
                if self.apply_funding_rate and symbol in self.funding_rate:
                    old_rate = self.funding_rate[symbol]
                    new_rate = old_rate * random.uniform(2, 5) * random.choice([-1, 1])
                    self.funding_rate[symbol] = new_rate
                    
                    self.logger.info(f"Sự kiện ngẫu nhiên: Thay đổi funding rate {symbol} từ {old_rate:.6f} sang {new_rate:.6f}")
                    
            elif event_type == "market_halt":
                # Tạm dừng thị trường
                if symbol in self.current_prices:
                    # Đánh dấu bằng cách đặt giá về None
                    self.current_prices[symbol] = None
                    
                    self.logger.info(f"Sự kiện ngẫu nhiên: Tạm dừng thị trường {symbol}")
    
    def _save_state(self) -> None:
        """
        Lưu trạng thái hiện tại của mô phỏng.
        """
        # Lưu lịch sử số dư
        self.balance_history.append({
            "timestamp": self.current_timestamp,
            "balance": self.current_balance,
            "total_value": self._calculate_total_value()
        })
        
        # Lưu lịch sử danh mục
        portfolio_snapshot = {
            "timestamp": self.current_timestamp,
            "balance": self.current_balance,
            "positions": deepcopy(self.open_positions),
            "prices": deepcopy(self.current_prices),
            "unrealized_pnl": sum(pos["unrealized_pnl"] for pos in self.open_positions.values()),
            "realized_pnl": sum(exec["fee"] for exec in self.execution_history if exec["success"])
        }
        
        self.portfolio_history.append(portfolio_snapshot)
    
    def _calculate_total_value(self) -> float:
        """
        Tính tổng giá trị danh mục (số dư + giá trị vị thế).
        
        Returns:
            float: Tổng giá trị danh mục.
        """
        total_value = self.current_balance
        
        # Cộng giá trị của các vị thế
        for symbol, position in self.open_positions.items():
            if symbol in self.current_prices and self.current_prices[symbol] is not None:
                current_price = self.current_prices[symbol]["close"]
                position_value = position["size"] * current_price
                total_value += position_value
        
        return total_value
    
    def reset(self) -> Dict[str, Any]:
        """
        Đặt lại trạng thái mô phỏng về ban đầu.
        
        Returns:
            Dict[str, Any]: Trạng thái ban đầu.
        """
        # Đặt lại các biến trạng thái
        self.current_balance = self.initial_balance
        self.current_timestamp = None
        self.current_timestep = 0
        self.current_prices = {}
        self.current_orderbooks = {}
        self.open_positions = {}
        self.pending_orders = []
        self.execution_history = []
        self.balance_history = []
        self.portfolio_history = []
        
        # Đặt lại môi trường
        for env in self.environments.values():
            env.reset()
        
        # Trả về trạng thái ban đầu
        return {
            "balance": self.current_balance,
            "positions": self.open_positions,
            "timestamp": self.current_timestamp,
            "prices": self.current_prices
        }
    
    def step(self) -> Tuple[Dict[str, Any], bool]:
        """
        Tiến hành một bước mô phỏng.
        
        Returns:
            Tuple[Dict[str, Any], bool]: (Trạng thái mới, đã kết thúc hay chưa).
        """
        # Kiểm tra xem đã kết thúc mô phỏng chưa
        if self.end_date and (self.current_timestamp and self.current_timestamp >= self.end_date):
            return self._get_current_state(), True
        
        # Nếu chưa bắt đầu, thiết lập timestamp ban đầu
        if self.current_timestamp is None:
            first_timestamps = []
            
            for symbol, df in self.data.items():
                if not df.empty:
                    first_timestamps.append(df.index[0])
            
            if first_timestamps:
                self.current_timestamp = max(first_timestamps)
                self.current_timestep = 0
            else:
                self.logger.error("Không có dữ liệu để bắt đầu mô phỏng")
                return self._get_current_state(), True
        else:
            # Cập nhật timestamp cho bước tiếp theo
            self._update_timestamp()
        
        # Kiểm tra lại xem đã kết thúc mô phỏng chưa
        if self.end_date and self.current_timestamp >= self.end_date:
            return self._get_current_state(), True
        
        # Cập nhật giá hiện tại cho các symbol
        self._update_prices()
        
        # Tạo sự kiện ngẫu nhiên
        self._generate_random_events()
        
        # Tính phí funding nếu có
        self._calculate_funding_fee()
        
        # Xử lý các lệnh đang chờ
        self._process_pending_orders()
        
        # Cập nhật trạng thái các vị thế
        self._update_positions()
        
        # Lưu trạng thái
        self._save_state()
        
        # Tăng bước mô phỏng
        self.current_timestep += 1
        
        # Trả về trạng thái hiện tại
        return self._get_current_state(), False
    
    def _update_timestamp(self) -> None:
        """
        Cập nhật timestamp cho bước tiếp theo.
        """
        # Chuyển đổi từ chuỗi sang timedelta
        if self.timeframe.endswith("m"):
            minutes = int(self.timeframe[:-1])
            delta = timedelta(minutes=minutes)
        elif self.timeframe.endswith("h"):
            hours = int(self.timeframe[:-1])
            delta = timedelta(hours=hours)
        elif self.timeframe.endswith("d"):
            days = int(self.timeframe[:-1])
            delta = timedelta(days=days)
        elif self.timeframe.endswith("w"):
            weeks = int(self.timeframe[:-1])
            delta = timedelta(weeks=weeks)
        else:
            # Mặc định là 1 giờ
            delta = timedelta(hours=1)
        
        # Cập nhật timestamp
        self.current_timestamp += delta
    
    def _update_prices(self) -> None:
        """
        Cập nhật giá hiện tại cho các symbol.
        """
        for symbol in self.symbols:
            try:
                # Lấy dữ liệu gần với timestamp hiện tại nhất
                df = self.data.get(symbol)
                
                if df is None or df.empty:
                    continue
                
                # Tìm điểm dữ liệu gần nhất
                closest_idx = df.index.get_indexer([self.current_timestamp], method="nearest")[0]
                
                if closest_idx < 0 or closest_idx >= len(df):
                    continue
                
                closest_timestamp = df.index[closest_idx]
                
                # Kiểm tra xem timestamp có quá xa không
                time_diff = abs((closest_timestamp - self.current_timestamp).total_seconds())
                allowed_diff = self._get_timeframe_seconds() * 1.5  # Cho phép sai lệch 1.5 lần khung thời gian
                
                if time_diff > allowed_diff:
                    self.logger.debug(f"Timestamp cho {symbol} quá xa: {closest_timestamp} vs {self.current_timestamp} ({time_diff}s)")
                    self.current_prices[symbol] = None
                    continue
                
                # Lấy dữ liệu OHLCV
                row = df.iloc[closest_idx]
                
                price_data = {
                    "timestamp": closest_timestamp,
                    "open": row["open"] if "open" in row else None,
                    "high": row["high"] if "high" in row else None,
                    "low": row["low"] if "low" in row else None,
                    "close": row["close"] if "close" in row else None,
                    "volume": row["volume"] if "volume" in row else None
                }
                
                # Cập nhật giá hiện tại
                self.current_prices[symbol] = price_data
                
                # Cập nhật orderbook nếu có dữ liệu
                if all(k in row for k in ["ask_price", "ask_volume", "bid_price", "bid_volume"]):
                    self.current_orderbooks[symbol] = {
                        "asks": [(row["ask_price"], row["ask_volume"])],
                        "bids": [(row["bid_price"], row["bid_volume"])]
                    }
                elif all(k in row for k in ["asks", "bids"]):
                    self.current_orderbooks[symbol] = {
                        "asks": json.loads(row["asks"]) if isinstance(row["asks"], str) else row["asks"],
                        "bids": json.loads(row["bids"]) if isinstance(row["bids"], str) else row["bids"]
                    }
                else:
                    # Tạo orderbook giả lập từ OHLC
                    mid_price = price_data["close"]
                    spread_pct = self.spread or 0.001  # Mặc định 0.1%
                    
                    ask_price = mid_price * (1 + spread_pct/2)
                    bid_price = mid_price * (1 - spread_pct/2)
                    
                    # Khối lượng giả lập từ volume
                    volume = price_data["volume"] or 1.0
                    
                    self.current_orderbooks[symbol] = {
                        "asks": [(ask_price, volume * 0.5)],
                        "bids": [(bid_price, volume * 0.5)]
                    }
                    
            except Exception as e:
                self.logger.error(f"Lỗi khi cập nhật giá cho {symbol}: {str(e)}")
                self.current_prices[symbol] = None
    
    def _get_timeframe_seconds(self) -> int:
        """
        Chuyển đổi timeframe sang số giây.
        
        Returns:
            int: Số giây tương ứng với timeframe.
        """
        if self.timeframe.endswith("m"):
            return int(self.timeframe[:-1]) * 60
        elif self.timeframe.endswith("h"):
            return int(self.timeframe[:-1]) * 3600
        elif self.timeframe.endswith("d"):
            return int(self.timeframe[:-1]) * 86400
        elif self.timeframe.endswith("w"):
            return int(self.timeframe[:-1]) * 604800
        else:
            # Mặc định là 1 giờ
            return 3600
    
    def _get_current_state(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của mô phỏng.
        
        Returns:
            Dict[str, Any]: Trạng thái hiện tại.
        """
        return {
            "timestamp": self.current_timestamp,
            "timestep": self.current_timestep,
            "balance": self.current_balance,
            "total_value": self._calculate_total_value(),
            "positions": self.open_positions,
            "prices": self.current_prices,
            "orderbooks": self.current_orderbooks,
            "pending_orders": self.pending_orders
        }
    
    def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Đặt lệnh giao dịch.
        
        Args:
            order (Dict[str, Any]): Thông tin lệnh giao dịch.
            
        Returns:
            Dict[str, Any]: Kết quả đặt lệnh.
        """
        # Kiểm tra các thông tin bắt buộc
        required_fields = ["symbol", "side", "type", "size"]
        missing_fields = [field for field in required_fields if field not in order]
        
        if missing_fields:
            return {
                "success": False,
                "reason": f"Thiếu các trường thông tin: {', '.join(missing_fields)}"
            }
        
        # Kiểm tra symbol
        if order["symbol"] not in self.symbols:
            return {
                "success": False,
                "reason": f"Symbol không hợp lệ: {order['symbol']}"
            }
        
        # Kiểm tra loại lệnh
        if order["type"] not in [OrderType.MARKET.value, OrderType.LIMIT.value]:
            return {
                "success": False,
                "reason": f"Loại lệnh không được hỗ trợ: {order['type']}"
            }
        
        # Kiểm tra giá nếu là lệnh limit
        if order["type"] == OrderType.LIMIT.value and "price" not in order:
            return {
                "success": False,
                "reason": f"Thiếu giá cho lệnh limit"
            }
        
        # Tạo ID cho lệnh nếu chưa có
        if "order_id" not in order:
            order["order_id"] = f"order_{len(self.execution_history) + len(self.pending_orders) + 1}_{self.current_timestamp.strftime('%Y%m%d%H%M%S')}"
        
        # Áp dụng độ trễ
        latency = self.latency_model()
        order["latency"] = latency
        order["timestamp"] = self.current_timestamp
        
        # Thêm vào hàng đợi hoặc thực thi ngay
        if latency > 0:
            # Thêm vào danh sách chờ, sẽ được xử lý sau
            self.pending_orders.append(order)
            
            return {
                "success": True,
                "order_id": order["order_id"],
                "status": "pending",
                "message": f"Lệnh đã được đặt với độ trễ {latency}s"
            }
        else:
            # Thực thi ngay
            if order["type"] == OrderType.MARKET.value:
                # Lệnh market được thực thi ngay
                result = self._execute_order(order)
                return result
            else:
                # Lệnh limit được thêm vào hàng đợi
                self.pending_orders.append(order)
                
                return {
                    "success": True,
                    "order_id": order["order_id"],
                    "status": "pending",
                    "message": f"Lệnh limit đã được đặt"
                }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Hủy lệnh đang chờ.
        
        Args:
            order_id (str): ID của lệnh cần hủy.
            
        Returns:
            Dict[str, Any]: Kết quả hủy lệnh.
        """
        # Tìm lệnh trong danh sách chờ
        for i, order in enumerate(self.pending_orders):
            if order["order_id"] == order_id:
                # Xóa khỏi danh sách
                self.pending_orders.pop(i)
                
                self.logger.info(f"Đã hủy lệnh {order_id}")
                
                return {
                    "success": True,
                    "order_id": order_id,
                    "message": f"Lệnh đã được hủy"
                }
        
        # Không tìm thấy lệnh
        return {
            "success": False,
            "order_id": order_id,
            "reason": f"Không tìm thấy lệnh {order_id} trong danh sách chờ"
        }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Lấy trạng thái của lệnh.
        
        Args:
            order_id (str): ID của lệnh cần kiểm tra.
            
        Returns:
            Dict[str, Any]: Trạng thái của lệnh.
        """
        # Kiểm tra trong danh sách lệnh đang chờ
        for order in self.pending_orders:
            if order["order_id"] == order_id:
                return {
                    "order_id": order_id,
                    "status": OrderStatus.PENDING.value,
                    "symbol": order["symbol"],
                    "side": order["side"],
                    "type": order["type"],
                    "size": order["size"],
                    "price": order.get("price"),
                    "timestamp": order["timestamp"]
                }
        
        # Kiểm tra trong lịch sử thực thi
        for execution in self.execution_history:
            if execution["order_id"] == order_id:
                status = OrderStatus.FILLED.value if execution["success"] else OrderStatus.REJECTED.value
                
                return {
                    "order_id": order_id,
                    "status": status,
                    "symbol": execution["symbol"],
                    "side": execution["side"],
                    "type": execution["type"],
                    "size": execution["requested_size"],
                    "filled_size": execution.get("filled_size"),
                    "price": execution.get("execution_price"),
                    "timestamp": execution["timestamp"],
                    "success": execution["success"],
                    "reason": execution.get("reason")
                }
        
        # Không tìm thấy lệnh
        return {
            "order_id": order_id,
            "status": OrderStatus.UNKNOWN.value,
            "reason": f"Không tìm thấy lệnh {order_id}"
        }
    
    def create_environment(self, symbol: str, window_size: int = 100, include_orderbook: bool = False, use_simple_env: bool = False) -> BaseEnvironment:
        """
        Tạo môi trường giao dịch cho symbol.
        
        Args:
            symbol (str): Symbol cần tạo môi trường.
            window_size (int, optional): Kích thước cửa sổ quan sát. Mặc định là 100.
            include_orderbook (bool, optional): Bao gồm dữ liệu orderbook trong môi trường. Mặc định là False.
            use_simple_env (bool, optional): Sử dụng môi trường đơn giản thay vì TradingEnv đầy đủ. Mặc định là False.
            
        Returns:
            BaseEnvironment: Môi trường giao dịch mới tạo.
        """
        # Kiểm tra dữ liệu
        if symbol not in self.data or self.data[symbol].empty:
            raise ValueError(f"Không có dữ liệu cho symbol {symbol}")
        
        # Tạo môi trường giao dịch
        env = TradingEnv(
            data=self.data[symbol],
            symbol=symbol,
            timeframe=self.timeframe,
            initial_balance=self.initial_balance,
            fee_rate=self.fee_rate,
            window_size=window_size,
            max_positions=1,  # Mỗi môi trường chỉ quản lý 1 vị thế
            include_timestamp=True,
            stoploss_percentage=None,
            takeprofit_percentage=None,
            logger=self.logger
        )
        
        # Lưu vào danh sách môi trường
        self.environments[symbol] = env
        
        # Reset môi trường
        env.reset()
        
        return env
    
    def get_equity_curve(self) -> pd.Series:
        """
        Lấy đường cong vốn từ lịch sử.
        
        Returns:
            pd.Series: Chuỗi giá trị vốn theo thời gian.
        """
        if not self.balance_history:
            return pd.Series()
        
        # Tạo chuỗi từ lịch sử
        timestamps = [entry["timestamp"] for entry in self.balance_history]
        values = [entry["total_value"] for entry in self.balance_history]
        
        return pd.Series(values, index=timestamps)
    
    def get_trade_history(self) -> pd.DataFrame:
        """
        Lấy lịch sử giao dịch.
        
        Returns:
            pd.DataFrame: DataFrame chứa lịch sử giao dịch.
        """
        if not self.execution_history:
            return pd.DataFrame()
        
        # Lọc các giao dịch thành công
        successful_executions = [exec for exec in self.execution_history if exec["success"]]
        
        if not successful_executions:
            return pd.DataFrame()
        
        # Tạo DataFrame
        df = pd.DataFrame(successful_executions)
        
        # Thêm các cột phân tích
        if "symbol" in df.columns and "filled_size" in df.columns and "execution_price" in df.columns:
            df["volume"] = df["filled_size"] * df["execution_price"]
            
            # Tính P&L
            df["entry_price"] = None
            df["exit_price"] = None
            df["profit"] = None
            df["duration"] = None
            
            # Theo dõi giá vào cho mỗi symbol
            entry_prices = {}
            entry_times = {}
            
            for i, row in df.iterrows():
                symbol = row["symbol"]
                side = row["side"]
                price = row["execution_price"]
                timestamp = row["timestamp"]
                
                if side == "buy":
                    # Lưu giá vào
                    entry_prices[symbol] = price
                    entry_times[symbol] = timestamp
                    df.at[i, "entry_price"] = price
                else:  # sell
                    # Nếu có giá vào trước đó
                    if symbol in entry_prices:
                        entry_price = entry_prices[symbol]
                        entry_time = entry_times[symbol]
                        
                        # Tính P&L
                        profit = (price - entry_price) * row["filled_size"]
                        df.at[i, "entry_price"] = entry_price
                        df.at[i, "exit_price"] = price
                        df.at[i, "profit"] = profit
                        
                        # Tính thời gian nắm giữ
                        if entry_time is not None:
                            duration = (timestamp - entry_time).total_seconds() / 3600  # giờ
                            df.at[i, "duration"] = duration
                        
                        # Xóa giá vào đã sử dụng
                        del entry_prices[symbol]
                        del entry_times[symbol]
        
        return df
    
    def get_position_history(self) -> pd.DataFrame:
        """
        Lấy lịch sử vị thế.
        
        Returns:
            pd.DataFrame: DataFrame chứa lịch sử vị thế.
        """
        if not self.portfolio_history:
            return pd.DataFrame()
        
        # Tạo danh sách các bản ghi
        records = []
        
        for snapshot in self.portfolio_history:
            timestamp = snapshot["timestamp"]
            
            # Với mỗi vị thế trong snapshot
            for symbol, position in snapshot["positions"].items():
                record = {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": position["side"],
                    "size": position["size"],
                    "entry_price": position["entry_price"],
                    "current_price": snapshot["prices"].get(symbol, {}).get("close") if symbol in snapshot["prices"] else None,
                    "unrealized_pnl": position["unrealized_pnl"],
                    "entry_time": position["entry_time"]
                }
                
                records.append(record)
        
        # Tạo DataFrame
        if records:
            return pd.DataFrame(records)
        else:
            return pd.DataFrame()
    
    def run_simulation(self, steps: int = None, until_date: Union[str, datetime] = None) -> Dict[str, Any]:
        """
        Chạy mô phỏng một số bước hoặc đến ngày cụ thể.
        
        Args:
            steps (int, optional): Số bước cần chạy. Mặc định là None (chạy đến cuối).
            until_date (Union[str, datetime], optional): Ngày cần chạy đến. Mặc định là None (chạy đến cuối).
            
        Returns:
            Dict[str, Any]: Kết quả mô phỏng.
        """
        # Đặt lại trạng thái
        self.reset()
        
        # Chuyển đổi until_date sang datetime nếu cần
        if until_date is not None and not isinstance(until_date, datetime):
            until_date = self._parse_date(until_date)
        
        step_count = 0
        states = []
        
        # Lặp cho đến khi kết thúc mô phỏng
        while True:
            # Thực hiện một bước
            state, done = self.step()
            
            # Lưu trạng thái
            states.append(state)
            
            # Tăng số bước
            step_count += 1
            
            # Kiểm tra điều kiện dừng
            if done:
                break
                
            if steps is not None and step_count >= steps:
                break
                
            if until_date is not None and state["timestamp"] >= until_date:
                break
        
        # Tạo kết quả mô phỏng
        equity_curve = self.get_equity_curve()
        trade_history = self.get_trade_history()
        position_history = self.get_position_history()
        
        # Tính các chỉ số hiệu suất nếu có đủ dữ liệu
        performance_metrics = {}
        
        if len(equity_curve) > 1:
            from backtesting.performance_metrics import PerformanceMetrics
            
            pm = PerformanceMetrics(equity_curve=equity_curve, trades=trade_history)
            performance_metrics = pm.calculate_all_metrics()
        
        # Tổng hợp kết quả
        result = {
            "equity_curve": equity_curve,
            "trade_history": trade_history,
            "position_history": position_history,
            "performance_metrics": performance_metrics,
            "final_balance": self.current_balance,
            "total_value": self._calculate_total_value(),
            "open_positions": self.open_positions,
            "steps_executed": step_count,
            "final_timestamp": self.current_timestamp,
            "states": states
        }
        
        self.logger.info(f"Đã chạy mô phỏng {step_count} bước từ {states[0]['timestamp']} đến {self.current_timestamp}")
        self.logger.info(f"Kết quả: Số dư cuối: {self.current_balance:.2f}, Tổng giá trị: {self._calculate_total_value():.2f}")
        
        return result
    
    def backtest_strategy(self, strategy_fn: Callable, strategy_params: Dict[str, Any] = None, steps: int = None, until_date: Union[str, datetime] = None) -> Dict[str, Any]:
        """
        Chạy backtest với chiến lược nhất định.
        
        Args:
            strategy_fn (Callable): Hàm chiến lược (nhận state, simulator, params và trả về dict hành động).
            strategy_params (Dict[str, Any], optional): Tham số cho chiến lược. Mặc định là None.
            steps (int, optional): Số bước cần chạy. Mặc định là None (chạy đến cuối).
            until_date (Union[str, datetime], optional): Ngày cần chạy đến. Mặc định là None (chạy đến cuối).
            
        Returns:
            Dict[str, Any]: Kết quả mô phỏng.
        """
        # Đặt lại trạng thái
        self.reset()
        
        # Khởi tạo tham số chiến lược
        strategy_params = strategy_params or {}
        
        # Chuyển đổi until_date sang datetime nếu cần
        if until_date is not None and not isinstance(until_date, datetime):
            until_date = self._parse_date(until_date)
        
        step_count = 0
        states = []
        strategy_decisions = []
        
        # Lặp cho đến khi kết thúc mô phỏng
        while True:
            # Thực hiện một bước
            state, done = self.step()
            
            # Lưu trạng thái
            states.append(state)
            
            # Thực thi chiến lược
            try:
                decision = strategy_fn(state, self, strategy_params)
                strategy_decisions.append(decision)
                
                # Thực thi các hành động từ chiến lược
                if decision is not None and isinstance(decision, dict):
                    if "orders" in decision and isinstance(decision["orders"], list):
                        for order in decision["orders"]:
                            self.place_order(order)
                    
                    if "cancel_orders" in decision and isinstance(decision["cancel_orders"], list):
                        for order_id in decision["cancel_orders"]:
                            self.cancel_order(order_id)
            except Exception as e:
                self.logger.error(f"Lỗi khi thực thi chiến lược: {str(e)}")
            
            # Tăng số bước
            step_count += 1
            
            # Kiểm tra điều kiện dừng
            if done:
                break
                
            if steps is not None and step_count >= steps:
                break
                
            if until_date is not None and state["timestamp"] >= until_date:
                break
        
        # Tạo kết quả mô phỏng
        equity_curve = self.get_equity_curve()
        trade_history = self.get_trade_history()
        position_history = self.get_position_history()
        
        # Tính các chỉ số hiệu suất nếu có đủ dữ liệu
        performance_metrics = {}
        
        if len(equity_curve) > 1:
            from backtesting.performance_metrics import PerformanceMetrics
            
            pm = PerformanceMetrics(equity_curve=equity_curve, trades=trade_history)
            performance_metrics = pm.calculate_all_metrics()
        
        # Tổng hợp kết quả
        result = {
            "equity_curve": equity_curve,
            "trade_history": trade_history,
            "position_history": position_history,
            "performance_metrics": performance_metrics,
            "final_balance": self.current_balance,
            "total_value": self._calculate_total_value(),
            "open_positions": self.open_positions,
            "steps_executed": step_count,
            "final_timestamp": self.current_timestamp,
            "states": states,
            "strategy_decisions": strategy_decisions
        }
        
        self.logger.info(f"Đã chạy backtest chiến lược {step_count} bước từ {states[0]['timestamp']} đến {self.current_timestamp}")
        self.logger.info(f"Kết quả: Số dư cuối: {self.current_balance:.2f}, Tổng giá trị: {self._calculate_total_value():.2f}")
        
        return result
    
    def walk_forward_analysis(self, strategy_fn: Callable, strategy_params: Dict[str, Any], window_size: int, step_size: int, optimization_fn: Callable = None) -> Dict[str, Any]:
        """
        Thực hiện phân tích walk-forward.
        
        Args:
            strategy_fn (Callable): Hàm chiến lược.
            strategy_params (Dict[str, Any]): Tham số ban đầu cho chiến lược.
            window_size (int): Kích thước cửa sổ cho mỗi phân tích (số bước).
            step_size (int): Kích thước bước giữa các cửa sổ (số bước).
            optimization_fn (Callable, optional): Hàm tối ưu hóa tham số. Mặc định là None.
            
        Returns:
            Dict[str, Any]: Kết quả phân tích.
        """
        # Đặt lại trạng thái
        self.reset()
        
        # Tạo danh sách cửa sổ
        all_dates = []
        
        for symbol, df in self.data.items():
            all_dates.extend(df.index.tolist())
        
        all_dates = sorted(list(set(all_dates)))
        
        if not all_dates:
            self.logger.error("Không có dữ liệu để phân tích")
            return {}
        
        # Tạo các cửa sổ
        windows = []
        
        for i in range(0, len(all_dates) - window_size, step_size):
            in_sample_start = all_dates[i]
            in_sample_end = all_dates[i + window_size // 2]
            out_sample_start = all_dates[i + window_size // 2 + 1]
            out_sample_end = all_dates[i + window_size]
            
            windows.append({
                "in_sample": (in_sample_start, in_sample_end),
                "out_sample": (out_sample_start, out_sample_end)
            })
        
        # Thực hiện phân tích cho từng cửa sổ
        results = []
        
        for i, window in enumerate(windows):
            self.logger.info(f"Phân tích cửa sổ {i+1}/{len(windows)}: In-sample {window['in_sample'][0]} - {window['in_sample'][1]}, Out-of-sample {window['out_sample'][0]} - {window['out_sample'][1]}")
            
            # Tối ưu hóa tham số trên dữ liệu in-sample
            optimized_params = strategy_params
            
            if optimization_fn is not None:
                # Tạo một bản sao để tối ưu hóa
                simulator_copy = deepcopy(self)
                simulator_copy.reset()
                
                # Thực hiện tối ưu hóa
                optimized_params = optimization_fn(
                    simulator_copy, 
                    strategy_fn, 
                    strategy_params,
                    window["in_sample"][0],
                    window["in_sample"][1]
                )
            
            # Đặt lại trạng thái
            self.reset()
            
            # Chạy backtest trên dữ liệu out-of-sample với tham số đã tối ưu
            result = self.backtest_strategy(
                strategy_fn,
                optimized_params,
                until_date=window["out_sample"][1]
            )
            
            # Thêm thông tin cửa sổ
            result["window"] = window
            result["window_index"] = i
            result["optimized_params"] = optimized_params
            
            # Thêm vào kết quả
            results.append(result)
        
        # Tổng hợp kết quả
        combined_equity = []
        combined_trades = []
        
        for result in results:
            # Thêm equity curve
            combined_equity.append(result["equity_curve"])
            
            # Thêm giao dịch
            if not result["trade_history"].empty:
                combined_trades.append(result["trade_history"])
        
        # Kết hợp equity curve
        combined_equity_curve = pd.concat(combined_equity) if combined_equity else pd.Series()
        
        # Kết hợp lịch sử giao dịch
        combined_trade_history = pd.concat(combined_trades) if combined_trades else pd.DataFrame()
        
        # Tính các chỉ số hiệu suất cho toàn bộ phân tích
        performance_metrics = {}
        
        if len(combined_equity_curve) > 1:
            from backtesting.performance_metrics import PerformanceMetrics
            
            pm = PerformanceMetrics(equity_curve=combined_equity_curve, trades=combined_trade_history)
            performance_metrics = pm.calculate_all_metrics()
        
        # Tính biến động tham số
        param_stability = {}
        
        if len(results) > 1:
            for param_name in results[0]["optimized_params"].keys():
                values = [result["optimized_params"].get(param_name) for result in results]
                
                if all(isinstance(v, (int, float)) for v in values):
                    param_stability[param_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": min(values),
                        "max": max(values),
                        "values": values
                    }
        
        # Trả về kết quả tổng hợp
        return {
            "windows": windows,
            "window_results": results,
            "combined_equity_curve": combined_equity_curve,
            "combined_trade_history": combined_trade_history,
            "performance_metrics": performance_metrics,
            "param_stability": param_stability
        }
    
    def monte_carlo_analysis(self, strategy_fn: Callable, strategy_params: Dict[str, Any], num_simulations: int = 30, randomize_fn: Callable = None) -> Dict[str, Any]:
        """
        Thực hiện phân tích Monte Carlo.
        
        Args:
            strategy_fn (Callable): Hàm chiến lược.
            strategy_params (Dict[str, Any]): Tham số cho chiến lược.
            num_simulations (int, optional): Số lần mô phỏng. Mặc định là 30.
            randomize_fn (Callable, optional): Hàm ngẫu nhiên hóa. Mặc định là None.
            
        Returns:
            Dict[str, Any]: Kết quả phân tích.
        """
        # Hàm ngẫu nhiên hóa mặc định
        def default_randomize(simulator, params):
            # Tạo seed ngẫu nhiên
            new_seed = np.random.randint(0, 2**32 - 1)
            np.random.seed(new_seed)
            random.seed(new_seed)
            
            # Ngẫu nhiên hóa tham số
            if 'slippage' in params:
                params['slippage'] = params['slippage'] * np.random.uniform(0.8, 1.2)
            
            if 'fee_rate' in params:
                params['fee_rate'] = params['fee_rate'] * np.random.uniform(0.9, 1.1)
            
            # Trả về tham số đã ngẫu nhiên hóa
            return params
        
        # Sử dụng hàm ngẫu nhiên mặc định nếu không được cung cấp
        randomize_fn = randomize_fn or default_randomize
        
        # Thực hiện các lần mô phỏng
        simulation_results = []
        
        for i in range(num_simulations):
            self.logger.info(f"Chạy mô phỏng Monte Carlo {i+1}/{num_simulations}")
            
            # Tạo một bản sao để mô phỏng
            simulator_copy = deepcopy(self)
            simulator_copy.reset()
            
            # Ngẫu nhiên hóa tham số
            randomized_params = randomize_fn(simulator_copy, deepcopy(strategy_params))
            
            # Chạy backtest
            result = simulator_copy.backtest_strategy(strategy_fn, randomized_params)
            
            # Thêm thông tin mô phỏng
            result["simulation_index"] = i
            result["randomized_params"] = randomized_params
            
            # Thêm vào kết quả
            simulation_results.append(result)
        
        # Phân tích kết quả
        equity_curves = [result["equity_curve"] for result in simulation_results]
        final_balances = [result["final_balance"] for result in simulation_results]
        
        # Tính thống kê
        percentiles = {
            "5%": np.percentile(final_balances, 5),
            "25%": np.percentile(final_balances, 25),
            "50%": np.percentile(final_balances, 50),
            "75%": np.percentile(final_balances, 75),
            "95%": np.percentile(final_balances, 95)
        }
        
        # Tạo DataFrame chứa tất cả equity curve
        all_equity = pd.DataFrame()
        
        for i, curve in enumerate(equity_curves):
            if not curve.empty:
                all_equity[f"sim_{i}"] = curve
        
        # Tính các đường phân vị
        percentile_curves = {}
        
        if not all_equity.empty:
            # Sắp xếp chỉ số
            all_equity.sort_index(inplace=True)
            
            # Tính các đường phân vị
            percentile_curves = {
                "5%": all_equity.quantile(0.05, axis=1),
                "25%": all_equity.quantile(0.25, axis=1),
                "median": all_equity.quantile(0.5, axis=1),
                "75%": all_equity.quantile(0.75, axis=1),
                "95%": all_equity.quantile(0.95, axis=1)
            }
        
        # Trả về kết quả phân tích
        return {
            "simulation_results": simulation_results,
            "final_balances": final_balances,
            "percentiles": percentiles,
            "equity_curves": equity_curves,
            "all_equity": all_equity,
            "percentile_curves": percentile_curves,
            "num_simulations": num_simulations,
            "mean_final_balance": np.mean(final_balances),
            "std_final_balance": np.std(final_balances),
            "worst_case": min(final_balances),
            "best_case": max(final_balances)
        }
    
    def parameter_sweep(self, strategy_fn: Callable, param_grid: Dict[str, List[Any]], fixed_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Thực hiện quét tham số để tìm tham số tối ưu.
        
        Args:
            strategy_fn (Callable): Hàm chiến lược.
            param_grid (Dict[str, List[Any]]): Lưới tham số cần quét.
            fixed_params (Dict[str, Any], optional): Tham số cố định. Mặc định là None.
            
        Returns:
            Dict[str, Any]: Kết quả quét tham số.
        """
        # Kết hợp tham số cố định
        fixed_params = fixed_params or {}
        
        # Tạo tất cả các tổ hợp tham số
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        combinations = list(product(*param_values))
        
        # Log số lượng tổ hợp
        self.logger.info(f"Quét tham số với {len(combinations)} tổ hợp")
        
        # Thực hiện backtest cho từng tổ hợp
        results = []
        
        for i, combination in enumerate(combinations):
            # Tạo bộ tham số
            params = fixed_params.copy()
            
            for j, name in enumerate(param_names):
                params[name] = combination[j]
            
            self.logger.info(f"Chạy tổ hợp {i+1}/{len(combinations)}: {params}")
            
            # Đặt lại trạng thái
            self.reset()
            
            # Chạy backtest
            result = self.backtest_strategy(strategy_fn, params)
            
            # Thêm thông tin tham số
            result["params"] = params
            result["combination_index"] = i
            
            # Trích xuất chỉ số quan trọng
            metrics = {
                "final_balance": result["final_balance"],
                "total_value": result["total_value"],
                "sharpe_ratio": result["performance_metrics"].get("sharpe_ratio", 0),
                "max_drawdown": result["performance_metrics"].get("max_drawdown", 0),
                "profit_factor": result["performance_metrics"].get("profit_factor", 0),
                "win_rate": result["performance_metrics"].get("win_rate", 0)
            }
            
            # Thêm vào kết quả
            result["metrics"] = metrics
            results.append(result)
        
        # Sắp xếp kết quả theo sharpe ratio
        sorted_results = sorted(results, key=lambda x: x["metrics"].get("sharpe_ratio", 0), reverse=True)
        
        # Tìm tham số tối ưu
        best_params = sorted_results[0]["params"] if sorted_results else {}
        
        # Tạo bảng so sánh
        comparison_table = []
        
        for result in results:
            row = result["params"].copy()
            row.update(result["metrics"])
            comparison_table.append(row)
        
        # Trả về kết quả
        return {
            "results": results,
            "sorted_results": sorted_results,
            "best_params": best_params,
            "comparison_table": comparison_table,
            "param_grid": param_grid,
            "fixed_params": fixed_params
        }
    
    def save_results(self, results: Dict[str, Any], file_path: str) -> None:
        """
        Lưu kết quả mô phỏng vào file.
        
        Args:
            results (Dict[str, Any]): Kết quả mô phỏng.
            file_path (str): Đường dẫn file lưu kết quả.
        """
        # Tạo một bản sao để lưu
        save_data = {}
        
        # Lưu các thông tin quan trọng
        if "equity_curve" in results:
            save_data["equity_curve"] = results["equity_curve"].to_dict() if isinstance(results["equity_curve"], pd.Series) else None
        
        if "trade_history" in results:
            save_data["trade_history"] = results["trade_history"].to_dict('records') if isinstance(results["trade_history"], pd.DataFrame) else None
        
        if "position_history" in results:
            save_data["position_history"] = results["position_history"].to_dict('records') if isinstance(results["position_history"], pd.DataFrame) else None
        
        if "performance_metrics" in results:
            save_data["performance_metrics"] = results["performance_metrics"]
        
        save_data["final_balance"] = results.get("final_balance")
        save_data["total_value"] = results.get("total_value")
        save_data["steps_executed"] = results.get("steps_executed")
        save_data["final_timestamp"] = results.get("final_timestamp").isoformat() if results.get("final_timestamp") else None
        
        # Lưu vào file
        try:
            # Xác định định dạng file
            if file_path.endswith(".json"):
                import json
                with open(file_path, 'w') as f:
                    json.dump(save_data, f, indent=2, default=str)
            elif file_path.endswith(".pickle") or file_path.endswith(".pkl"):
                import pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(save_data, f)
            else:
                # Mặc định là JSON
                import json
                with open(file_path, 'w') as f:
                    json.dump(save_data, f, indent=2, default=str)
            
            self.logger.info(f"Đã lưu kết quả vào {file_path}")
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu kết quả: {str(e)}")
    
    def load_results(self, file_path: str) -> Dict[str, Any]:
        """
        Tải kết quả mô phỏng từ file.
        
        Args:
            file_path (str): Đường dẫn file kết quả.
            
        Returns:
            Dict[str, Any]: Kết quả mô phỏng.
        """
        try:
            # Xác định định dạng file
            if file_path.endswith(".json"):
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif file_path.endswith(".pickle") or file_path.endswith(".pkl"):
                import pickle
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                # Mặc định là JSON
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
            
            # Chuyển đổi lại sang DataFrame/Series
            if "equity_curve" in data and data["equity_curve"]:
                data["equity_curve"] = pd.Series(data["equity_curve"])
            
            if "trade_history" in data and data["trade_history"]:
                data["trade_history"] = pd.DataFrame(data["trade_history"])
            
            if "position_history" in data and data["position_history"]:
                data["position_history"] = pd.DataFrame(data["position_history"])
            
            # Chuyển đổi timestamp
            if "final_timestamp" in data and data["final_timestamp"]:
                try:
                    data["final_timestamp"] = datetime.fromisoformat(data["final_timestamp"])
                except:
                    pass
            
            self.logger.info(f"Đã tải kết quả từ {file_path}")
            
            return data
        except Exception as e:
            self.logger.error(f"Lỗi khi tải kết quả: {str(e)}")
            return {}


# Example usage
if __name__ == "__main__":
    # Tạo dữ liệu giả lập cho ví dụ
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="1h")
    
    data = {
        "BTC/USDT": pd.DataFrame({
            "open": np.random.normal(20000, 1000, len(dates)),
            "high": np.random.normal(20500, 1000, len(dates)),
            "low": np.random.normal(19500, 1000, len(dates)),
            "close": np.random.normal(20200, 1000, len(dates)),
            "volume": np.random.normal(100, 30, len(dates))
        }, index=dates)
    }
    
    # Chiến lược đơn giản
    def simple_strategy(state, simulator, params):
        # Mua khi giá tăng
        actions = {
            "orders": []
        }
        
        for symbol, price_data in state["prices"].items():
            if price_data is None:
                continue
                
            current_price = price_data["close"]
            
            # Kiểm tra trong vị thế
            in_position = symbol in state["positions"]
            
            # Quyết định hành động
            if not in_position and len(state["positions"]) < params.get("max_positions", 1):
                # Tính % thay đổi giá gần đây
                prev_prices = [s["prices"].get(symbol, {}).get("close") for s in simulator.states[-params.get("lookback", 5):]]
                prev_prices = [p for p in prev_prices if p is not None]
                
                if len(prev_prices) >= params.get("lookback", 5):
                    price_change = (current_price / prev_prices[0] - 1) * 100
                    
                    if price_change > params.get("buy_threshold", 1.0):
                        # Tạo lệnh mua
                        order = {
                            "symbol": symbol,
                            "type": "MARKET",
                            "side": "buy",
                            "size": params.get("position_size", 0.1) * simulator.current_balance / current_price
                        }
                        
                        actions["orders"].append(order)
            elif in_position:
                position = state["positions"][symbol]
                unrealized_pnl = position["unrealized_pnl"]
                entry_price = position["entry_price"]
                
                # Tính % lợi nhuận
                profit_pct = (current_price / entry_price - 1) * 100 if position["side"] == "LONG" else (entry_price / current_price - 1) * 100
                
                # Đóng vị thế khi đạt ngưỡng lợi nhuận hoặc lỗ
                if profit_pct > params.get("take_profit", 3.0) or profit_pct < -params.get("stop_loss", 2.0):
                    order = {
                        "symbol": symbol,
                        "type": "MARKET",
                        "side": "sell",
                        "size": position["size"]
                    }
                    
                    actions["orders"].append(order)
        
        return actions
    
    # Tạo simulator
    simulator = HistoricalSimulator(
        data=data,
        timeframe="1h",
        initial_balance=10000.0,
        fee_rate=0.001,
        slippage=0.0005
    )
    
    # Tham số chiến lược
    strategy_params = {
        "lookback": 5,
        "buy_threshold": 1.0,
        "take_profit": 3.0,
        "stop_loss": 2.0,
        "position_size": 0.1,
        "max_positions": 1
    }
    
    # Chạy backtest
    results = simulator.backtest_strategy(simple_strategy, strategy_params)
    
    # Lưu kết quả
    simulator.save_results(results, "backtest_results.json")
    
    # Kiểm tra hiệu suất
    print(f"Final Balance: {results['final_balance']}")
    print(f"Total Value: {results['total_value']}")
    
    # Vẽ đường cong vốn
    if not results['equity_curve'].empty:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(results['equity_curve'])
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.savefig('equity_curve.png')

