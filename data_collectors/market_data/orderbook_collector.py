"""
Thu thập và phân tích dữ liệu orderbook từ các sàn giao dịch.
File này cung cấp các lớp và phương thức để thu thập, lưu trữ, và phân tích 
dữ liệu sổ lệnh (orderbook) từ các sàn giao dịch tiền điện tử, hỗ trợ
xây dựng bản đồ thanh khoản và tính toán các chỉ số liên quan đến thanh khoản.
"""

import os
import time
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Set, Deque
import logging
import json
from pathlib import Path
import collections
import threading
import heapq
from copy import deepcopy
import re

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_collectors.exchange_api.generic_connector import GenericExchangeConnector
from data_collectors.exchange_api.binance_connector import BinanceConnector
from data_collectors.exchange_api.bybit_connector import BybitConnector
from config.logging_config import setup_logger
from config.constants import Timeframe, TIMEFRAME_TO_SECONDS, Exchange, ErrorCode
from config.env import get_env
from config.system_config import DATA_DIR, BASE_DIR
from config.utils.validators import is_valid_trading_pair

class OrderbookSnapshot:
    """
    Lớp đại diện cho một snapshot của sổ lệnh tại một thời điểm.
    """
    
    def __init__(
        self,
        symbol: str,
        timestamp: Optional[int] = None,
        bids: Optional[List[List[float]]] = None,
        asks: Optional[List[List[float]]] = None,
        last_update_id: Optional[int] = None
    ):
        """
        Khởi tạo một snapshot orderbook.
        
        Args:
            symbol: Cặp giao dịch
            timestamp: Thời gian snapshot (ms)
            bids: Danh sách lệnh mua [[giá, khối lượng], ...]
            asks: Danh sách lệnh bán [[giá, khối lượng], ...]
            last_update_id: ID cập nhật cuối cùng
        """
        # Kiểm tra tính hợp lệ của symbol
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol không hợp lệ")
            
        self.symbol = symbol
        self.timestamp = timestamp or int(time.time() * 1000)
        self.bids = bids or []
        self.asks = asks or []
        self.last_update_id = last_update_id
        
        # Kiểm tra định dạng của bids và asks
        self._validate_orders()
        
        # Sắp xếp bids (giảm dần theo giá) và asks (tăng dần theo giá)
        self._sort_orders()
    
    def _validate_orders(self) -> None:
        """Kiểm tra tính hợp lệ của các lệnh."""
        # Kiểm tra bids
        for i, bid in enumerate(self.bids):
            if not isinstance(bid, list) or len(bid) < 2:
                raise ValueError(f"Định dạng không hợp lệ cho bid tại vị trí {i}: {bid}")
            
            # Chuyển đổi giá và khối lượng sang float nếu cần
            try:
                self.bids[i] = [float(bid[0]), float(bid[1])]
            except (ValueError, TypeError):
                raise ValueError(f"Không thể chuyển đổi giá hoặc khối lượng sang float cho bid: {bid}")
        
        # Kiểm tra asks
        for i, ask in enumerate(self.asks):
            if not isinstance(ask, list) or len(ask) < 2:
                raise ValueError(f"Định dạng không hợp lệ cho ask tại vị trí {i}: {ask}")
            
            # Chuyển đổi giá và khối lượng sang float nếu cần
            try:
                self.asks[i] = [float(ask[0]), float(ask[1])]
            except (ValueError, TypeError):
                raise ValueError(f"Không thể chuyển đổi giá hoặc khối lượng sang float cho ask: {ask}")
    
    def _sort_orders(self) -> None:
        """Sắp xếp các lệnh theo giá."""
        self.bids = sorted(self.bids, key=lambda x: float(x[0]), reverse=True)
        self.asks = sorted(self.asks, key=lambda x: float(x[0]))
    
    def get_mid_price(self) -> float:
        """
        Lấy giá trung bình giữa bid cao nhất và ask thấp nhất.
        
        Returns:
            Giá trung bình
        """
        if not self.bids or not self.asks:
            return 0.0
        
        best_bid = float(self.bids[0][0])
        best_ask = float(self.asks[0][0])
        
        return (best_bid + best_ask) / 2
    
    def get_spread(self) -> float:
        """
        Lấy spread (chênh lệch giữa bid cao nhất và ask thấp nhất).
        
        Returns:
            Spread
        """
        if not self.bids or not self.asks:
            return 0.0
        
        best_bid = float(self.bids[0][0])
        best_ask = float(self.asks[0][0])
        
        return best_ask - best_bid
    
    def get_spread_percentage(self) -> float:
        """
        Lấy spread dưới dạng phần trăm của giá trung bình.
        
        Returns:
            Spread percentage
        """
        mid_price = self.get_mid_price()
        if mid_price == 0:
            return 0.0
        
        spread = self.get_spread()
        return (spread / mid_price) * 100
    
    def get_liquidity_within_range(self, percentage: float = 1.0) -> Dict[str, float]:
        """
        Tính toán thanh khoản trong một khoảng giá nhất định.
        
        Args:
            percentage: Phần trăm khoảng giá (từ giá trung bình)
            
        Returns:
            Dict với khối lượng và giá trị cho bids và asks
        """
        # Kiểm tra giá trị percentage
        if percentage <= 0:
            raise ValueError("Percentage phải là số dương")
            
        mid_price = self.get_mid_price()
        if mid_price == 0:
            return {
                'bid_volume': 0.0,
                'ask_volume': 0.0,
                'bid_value': 0.0,
                'ask_value': 0.0
            }
        
        # Tính khoảng giá
        min_price = mid_price * (1 - percentage / 100)
        max_price = mid_price * (1 + percentage / 100)
        
        # Tính tổng khối lượng và giá trị
        bid_volume = sum(float(bid[1]) for bid in self.bids if float(bid[0]) >= min_price)
        ask_volume = sum(float(ask[1]) for ask in self.asks if float(ask[0]) <= max_price)
        
        bid_value = sum(float(bid[0]) * float(bid[1]) for bid in self.bids if float(bid[0]) >= min_price)
        ask_value = sum(float(ask[0]) * float(ask[1]) for ask in self.asks if float(ask[0]) <= max_price)
        
        return {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'bid_value': bid_value,
            'ask_value': ask_value
        }
    
    def get_imbalance(self) -> float:
        """
        Tính toán mất cân bằng giữa bên mua và bên bán.
        
        Returns:
            Mất cân bằng (-1.0 đến 1.0, dương là thiên về mua)
        """
        liquidity = self.get_liquidity_within_range(1.0)
        
        total_volume = liquidity['bid_volume'] + liquidity['ask_volume']
        if total_volume == 0:
            return 0.0
        
        # Mất cân bằng từ -1.0 đến 1.0
        return (liquidity['bid_volume'] - liquidity['ask_volume']) / total_volume
    
    def get_depth(self, levels: int = 10) -> Dict[str, List[List[float]]]:
        """
        Lấy độ sâu của sổ lệnh.
        
        Args:
            levels: Số cấp độ giá
            
        Returns:
            Dict với bids và asks
        """
        # Kiểm tra tham số levels
        if levels <= 0:
            raise ValueError("Levels phải là số dương")
            
        return {
            'bids': self.bids[:levels],
            'asks': self.asks[:levels]
        }
    
    def to_dict(self) -> Dict:
        """
        Chuyển đổi snapshot thành dict.
        
        Returns:
            Dict đại diện cho snapshot
        """
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp / 1000).isoformat(),
            'bids': self.bids,
            'asks': self.asks,
            'last_update_id': self.last_update_id,
            'mid_price': self.get_mid_price(),
            'spread': self.get_spread(),
            'spread_percentage': self.get_spread_percentage(),
            'imbalance': self.get_imbalance()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OrderbookSnapshot':
        """
        Tạo snapshot từ dict.
        
        Args:
            data: Dict chứa dữ liệu snapshot
            
        Returns:
            Instance của OrderbookSnapshot
        """
        # Kiểm tra các trường bắt buộc
        if 'symbol' not in data:
            raise ValueError("Thiếu trường 'symbol' trong dữ liệu")
            
        return cls(
            symbol=data.get('symbol'),
            timestamp=data.get('timestamp'),
            bids=data.get('bids', []),
            asks=data.get('asks', []),
            last_update_id=data.get('last_update_id')
        )
    
    @classmethod
    def from_ccxt_format(cls, symbol: str, orderbook: Dict) -> 'OrderbookSnapshot':
        """
        Tạo snapshot từ định dạng của CCXT.
        
        Args:
            symbol: Cặp giao dịch
            orderbook: Dữ liệu orderbook từ CCXT
            
        Returns:
            Instance của OrderbookSnapshot
        """
        # Kiểm tra orderbook
        if not isinstance(orderbook, dict):
            raise ValueError("Orderbook phải là một dict")
            
        if 'bids' not in orderbook or 'asks' not in orderbook:
            raise ValueError("Orderbook phải chứa trường 'bids' và 'asks'")
            
        return cls(
            symbol=symbol,
            timestamp=orderbook.get('timestamp'),
            bids=orderbook.get('bids', []),
            asks=orderbook.get('asks', []),
            last_update_id=orderbook.get('nonce')
        )


class OrderbookManager:
    """
    Lớp quản lý sổ lệnh theo thời gian thực.
    """
    
    def __init__(
        self,
        symbol: str,
        max_depth: int = 100,
        buffer_size: int = 10
    ):
        """
        Khởi tạo OrderbookManager.
        
        Args:
            symbol: Cặp giao dịch
            max_depth: Độ sâu tối đa của sổ lệnh
            buffer_size: Kích thước buffer cho các snapshot gần đây
        """
        # Kiểm tra tham số đầu vào
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol không hợp lệ")
        
        if max_depth <= 0:
            raise ValueError("max_depth phải là số dương")
            
        if buffer_size <= 0:
            raise ValueError("buffer_size phải là số dương")
            
        self.symbol = symbol
        self.max_depth = max_depth
        self.buffer_size = buffer_size
        
        # Sổ lệnh hiện tại
        self.current_orderbook = OrderbookSnapshot(symbol=symbol)
        
        # Buffer để lưu trữ các snapshot
        self.orderbook_buffer = collections.deque(maxlen=buffer_size)
        
        # Để đảm bảo thread-safety
        self.lock = threading.RLock()
        
        # Trạng thái
        self.last_update_time = 0
        self.last_update_id = 0
        self.is_synced = False
        
        # Logger
        self.logger = setup_logger(f"orderbook_manager_{symbol}")
        
        self.logger.info(f"Đã khởi tạo OrderbookManager cho {symbol}")
    
    def update_from_snapshot(self, snapshot: OrderbookSnapshot) -> bool:
        """
        Cập nhật sổ lệnh từ một snapshot.
        
        Args:
            snapshot: Snapshot cần cập nhật
            
        Returns:
            True nếu cập nhật thành công, False nếu không
        """
        # Kiểm tra snapshot
        if not isinstance(snapshot, OrderbookSnapshot):
            self.logger.error(f"Snapshot không hợp lệ: {type(snapshot)}")
            return False
            
        if snapshot.symbol != self.symbol:
            self.logger.error(f"Symbol không khớp: {snapshot.symbol} != {self.symbol}")
            return False
            
        with self.lock:
            # Lưu snapshot hiện tại vào buffer
            self.orderbook_buffer.append(deepcopy(self.current_orderbook))
            
            # Cập nhật sổ lệnh hiện tại
            self.current_orderbook = snapshot
            
            # Cập nhật trạng thái
            self.last_update_time = snapshot.timestamp
            self.last_update_id = snapshot.last_update_id
            self.is_synced = True
            
            return True
    
    def update_from_delta(self, delta: Dict) -> bool:
        """
        Cập nhật sổ lệnh từ một delta (các thay đổi gia tăng).
        
        Args:
            delta: Dữ liệu delta
            
        Returns:
            True nếu cập nhật thành công, False nếu không
        """
        # Kiểm tra delta
        if not isinstance(delta, dict):
            self.logger.error(f"Delta không hợp lệ: {type(delta)}")
            return False
            
        with self.lock:
            if not self.is_synced:
                self.logger.warning(f"Không thể cập nhật delta: sổ lệnh chưa được đồng bộ")
                return False
            
            # Kiểm tra ID cập nhật
            delta_update_id = delta.get('lastUpdateId', delta.get('u', 0))
            if delta_update_id <= self.last_update_id:
                self.logger.debug(f"Bỏ qua delta cũ: {delta_update_id} <= {self.last_update_id}")
                return False
            
            # Lưu snapshot hiện tại vào buffer
            self.orderbook_buffer.append(deepcopy(self.current_orderbook))
            
            # Cập nhật bids
            bids_to_update = delta.get('bids', delta.get('b', []))
            for bid in bids_to_update:
                if not isinstance(bid, list) or len(bid) < 2:
                    self.logger.warning(f"Bỏ qua bid không hợp lệ: {bid}")
                    continue
                    
                try:
                    price, amount = float(bid[0]), float(bid[1])
                    self._update_price_level('bids', price, amount)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Không thể chuyển đổi giá hoặc khối lượng cho bid {bid}: {e}")
            
            # Cập nhật asks
            asks_to_update = delta.get('asks', delta.get('a', []))
            for ask in asks_to_update:
                if not isinstance(ask, list) or len(ask) < 2:
                    self.logger.warning(f"Bỏ qua ask không hợp lệ: {ask}")
                    continue
                    
                try:
                    price, amount = float(ask[0]), float(ask[1])
                    self._update_price_level('asks', price, amount)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Không thể chuyển đổi giá hoặc khối lượng cho ask {ask}: {e}")
            
            # Sắp xếp lại sổ lệnh
            self.current_orderbook._sort_orders()
            
            # Cập nhật trạng thái
            self.last_update_time = delta.get('timestamp', int(time.time() * 1000))
            self.last_update_id = delta_update_id
            
            return True
    
    def _update_price_level(self, side: str, price: float, amount: float) -> None:
        """
        Cập nhật một mức giá trong sổ lệnh.
        
        Args:
            side: 'bids' hoặc 'asks'
            price: Giá
            amount: Khối lượng (0 để xóa)
        """
        # Kiểm tra side
        if side not in ['bids', 'asks']:
            raise ValueError(f"Side không hợp lệ: {side}, phải là 'bids' hoặc 'asks'")
            
        # Lấy danh sách lệnh
        orders = getattr(self.current_orderbook, side)
        
        # Tìm và cập nhật lệnh
        for i, order in enumerate(orders):
            if float(order[0]) == price:
                if amount == 0:
                    # Xóa lệnh
                    orders.pop(i)
                else:
                    # Cập nhật khối lượng
                    orders[i] = [price, amount]
                return
        
        # Nếu không tìm thấy và khối lượng > 0, thêm mới
        if amount > 0:
            orders.append([price, amount])
            
            # Giới hạn số lượng lệnh
            if len(orders) > self.max_depth:
                # Sắp xếp và cắt bớt
                if side == 'bids':
                    orders.sort(key=lambda x: float(x[0]), reverse=True)
                    orders = orders[:self.max_depth]
                else:
                    orders.sort(key=lambda x: float(x[0]))
                    orders = orders[:self.max_depth]
                
                # Cập nhật lại danh sách
                setattr(self.current_orderbook, side, orders)
    
    def get_current_snapshot(self) -> OrderbookSnapshot:
        """
        Lấy snapshot hiện tại của sổ lệnh.
        
        Returns:
            Snapshot hiện tại
        """
        with self.lock:
            return deepcopy(self.current_orderbook)
    
    def get_recent_snapshots(self, count: int = None) -> List[OrderbookSnapshot]:
        """
        Lấy các snapshot gần đây.
        
        Args:
            count: Số lượng snapshot (None để lấy tất cả)
            
        Returns:
            Danh sách các snapshot
        """
        # Kiểm tra tham số count
        if count is not None and count <= 0:
            raise ValueError("count phải là số dương")
            
        with self.lock:
            if count is None:
                return list(self.orderbook_buffer)
            else:
                return list(self.orderbook_buffer)[-count:]
    
    def get_vwap(self, volume: float, side: str = 'asks') -> float:
        """
        Tính giá trung bình theo khối lượng cho một lượng khối lượng cụ thể.
        
        Args:
            volume: Khối lượng cần tính
            side: 'asks' để mua, 'bids' để bán
            
        Returns:
            Giá trung bình theo khối lượng
        """
        # Kiểm tra tham số đầu vào
        if volume <= 0:
            raise ValueError("volume phải là số dương")
            
        if side not in ['bids', 'asks']:
            raise ValueError(f"side không hợp lệ: {side}, phải là 'bids' hoặc 'asks'")
            
        with self.lock:
            orders = getattr(self.current_orderbook, side)
            
            if not orders:
                return 0.0
            
            total_volume = 0.0
            total_value = 0.0
            
            for price, amount in orders:
                price, amount = float(price), float(amount)
                
                if total_volume + amount >= volume:
                    # Lấy phần còn lại
                    remaining = volume - total_volume
                    total_value += price * remaining
                    total_volume += remaining
                    break
                else:
                    # Lấy toàn bộ
                    total_value += price * amount
                    total_volume += amount
            
            if total_volume == 0:
                return 0.0
            
            return total_value / total_volume
    
    def get_executed_price(self, volume: float, side: str = 'buy') -> float:
        """
        Mô phỏng giá thực thi cho một lệnh thị trường.
        
        Args:
            volume: Khối lượng cần thực thi
            side: 'buy' hoặc 'sell'
            
        Returns:
            Giá thực thi ước tính
        """
        # Kiểm tra tham số đầu vào
        if volume <= 0:
            raise ValueError("volume phải là số dương")
            
        if side not in ['buy', 'sell']:
            raise ValueError(f"side không hợp lệ: {side}, phải là 'buy' hoặc 'sell'")
            
        book_side = 'asks' if side == 'buy' else 'bids'
        return self.get_vwap(volume, book_side)
    
    def get_liquidity_histogram(self, bins: int = 10, range_percentage: float = 5.0) -> Dict:
        """
        Tạo histogram thanh khoản.
        
        Args:
            bins: Số lượng bins
            range_percentage: Khoảng giá (phần trăm từ giá trung bình)
            
        Returns:
            Dict với histogram cho bids và asks
        """
        # Kiểm tra tham số đầu vào
        if bins <= 0:
            raise ValueError("bins phải là số dương")
            
        if range_percentage <= 0:
            raise ValueError("range_percentage phải là số dương")
            
        with self.lock:
            mid_price = self.current_orderbook.get_mid_price()
            
            if mid_price == 0:
                return {
                    'bids': {'prices': [], 'volumes': []},
                    'asks': {'prices': [], 'volumes': []}
                }
            
            # Tính khoảng giá
            min_price = mid_price * (1 - range_percentage / 100)
            max_price = mid_price * (1 + range_percentage / 100)
            
            # Tạo bins cho bids và asks
            bid_bins = np.linspace(min_price, mid_price, bins + 1)
            ask_bins = np.linspace(mid_price, max_price, bins + 1)
            
            # Khởi tạo histogram
            bid_hist = np.zeros(bins)
            ask_hist = np.zeros(bins)
            
            # Tính histogram cho bids
            for price, amount in self.current_orderbook.bids:
                price, amount = float(price), float(amount)
                
                if price < min_price or price > mid_price:
                    continue
                
                # Tìm bin
                bin_index = np.digitize(price, bid_bins) - 1
                if 0 <= bin_index < bins:
                    bid_hist[bin_index] += amount
            
            # Tính histogram cho asks
            for price, amount in self.current_orderbook.asks:
                price, amount = float(price), float(amount)
                
                if price < mid_price or price > max_price:
                    continue
                
                # Tìm bin
                bin_index = np.digitize(price, ask_bins) - 1
                if 0 <= bin_index < bins:
                    ask_hist[bin_index] += amount
            
            # Tạo kết quả
            return {
                'bids': {
                    'prices': bid_bins[:-1].tolist(),
                    'volumes': bid_hist.tolist()
                },
                'asks': {
                    'prices': ask_bins[:-1].tolist(),
                    'volumes': ask_hist.tolist()
                }
            }
    
    def calculate_liquidity_metrics(self) -> Dict:
        """
        Tính toán các chỉ số thanh khoản.
        
        Returns:
            Dict với các chỉ số thanh khoản
        """
        with self.lock:
            mid_price = self.current_orderbook.get_mid_price()
            
            # Tính thanh khoản ở các khoảng
            liquidity_1pct = self.current_orderbook.get_liquidity_within_range(1.0)
            liquidity_2pct = self.current_orderbook.get_liquidity_within_range(2.0)
            liquidity_5pct = self.current_orderbook.get_liquidity_within_range(5.0)
            
            # Tính mất cân bằng
            bid_ask_ratio_1pct = liquidity_1pct['bid_volume'] / liquidity_1pct['ask_volume'] if liquidity_1pct['ask_volume'] else 0
            
            return {
                'symbol': self.symbol,
                'timestamp': self.last_update_time,
                'mid_price': mid_price,
                'spread': self.current_orderbook.get_spread(),
                'spread_percentage': self.current_orderbook.get_spread_percentage(),
                'bid_ask_ratio': bid_ask_ratio_1pct,
                'imbalance': self.current_orderbook.get_imbalance(),
                'liquidity_1pct': liquidity_1pct,
                'liquidity_2pct': liquidity_2pct,
                'liquidity_5pct': liquidity_5pct,
                'bid_volume_1pct': liquidity_1pct['bid_volume'],
                'ask_volume_1pct': liquidity_1pct['ask_volume'],
                'total_volume_1pct': liquidity_1pct['bid_volume'] + liquidity_1pct['ask_volume'],
                'is_synced': self.is_synced
            }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Chuyển đổi sổ lệnh hiện tại thành DataFrame.
        
        Returns:
            DataFrame với dữ liệu sổ lệnh
        """
        with self.lock:
            # Tạo DataFrame cho bids
            if self.current_orderbook.bids:
                bids_df = pd.DataFrame(self.current_orderbook.bids, columns=['price', 'amount'])
                bids_df['side'] = 'bid'
            else:
                bids_df = pd.DataFrame(columns=['price', 'amount', 'side'])
            
            # Tạo DataFrame cho asks
            if self.current_orderbook.asks:
                asks_df = pd.DataFrame(self.current_orderbook.asks, columns=['price', 'amount'])
                asks_df['side'] = 'ask'
            else:
                asks_df = pd.DataFrame(columns=['price', 'amount', 'side'])
            
            # Ghép lại
            df = pd.concat([bids_df, asks_df])
            
            # Thêm thông tin
            df['symbol'] = self.symbol
            df['timestamp'] = self.last_update_time
            df['datetime'] = pd.to_datetime(self.last_update_time, unit='ms')
            
            return df
    
    def reset(self) -> None:
        """
        Reset sổ lệnh.
        """
        with self.lock:
            self.current_orderbook = OrderbookSnapshot(symbol=self.symbol)
            self.orderbook_buffer.clear()
            self.last_update_time = 0
            self.last_update_id = 0
            self.is_synced = False
            
            self.logger.info(f"Đã reset sổ lệnh cho {self.symbol}")


class OrderbookCollector:
    """
    Lớp chính để thu thập và phân tích dữ liệu sổ lệnh từ các sàn giao dịch.
    """
    
    def __init__(
        self,
        exchange_connector: GenericExchangeConnector,
        data_dir: Path = None,
        snapshot_interval: int = 60,  # Lấy snapshot mỗi 60 giây
        max_depth: int = 100,
        buffer_size: int = 10,
        max_file_snapshots: int = 1000  # Số lượng snapshot tối đa trong 1 file
    ):
        """
        Khởi tạo bộ thu thập sổ lệnh.
        
        Args:
            exchange_connector: Kết nối với sàn giao dịch
            data_dir: Thư mục lưu trữ dữ liệu
            snapshot_interval: Khoảng thời gian giữa các snapshot (giây)
            max_depth: Độ sâu tối đa của sổ lệnh
            buffer_size: Kích thước buffer cho các snapshot gần đây
            max_file_snapshots: Số lượng snapshot tối đa trong 1 file
        """
        # Kiểm tra tham số đầu vào
        if not isinstance(exchange_connector, GenericExchangeConnector):
            raise ValueError("exchange_connector phải là instance của GenericExchangeConnector")
            
        if snapshot_interval <= 0:
            raise ValueError("snapshot_interval phải là số dương")
            
        if max_depth <= 0:
            raise ValueError("max_depth phải là số dương")
            
        if buffer_size <= 0:
            raise ValueError("buffer_size phải là số dương")
            
        if max_file_snapshots <= 0:
            raise ValueError("max_file_snapshots phải là số dương")
            
        self.exchange_connector = exchange_connector
        self.exchange_id = exchange_connector.exchange_id
        self.logger = setup_logger(f"orderbook_collector_{self.exchange_id}")
        
        # Thiết lập thư mục lưu trữ dữ liệu
        if data_dir is None:
            self.data_dir = DATA_DIR / 'orderbook' / self.exchange_id
        else:
            self.data_dir = data_dir / 'orderbook' / self.exchange_id
        
        # Tạo thư mục lưu trữ nếu chưa tồn tại
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Cấu hình
        self.snapshot_interval = snapshot_interval
        self.max_depth = max_depth
        self.buffer_size = buffer_size
        self.max_file_snapshots = max_file_snapshots
        
        # Quản lý sổ lệnh
        self.orderbook_managers = {}  # {symbol: OrderbookManager}
        
        # Trạng thái
        self.is_running = False
        self.tasks = []
        
        # Callbacks
        self._update_callbacks = []
        self._snapshot_callbacks = []
        
        # Snapshot buffers và counters cho việc lưu gộp
        self._snapshot_buffers = {}  # {symbol: [snapshots]}
        self._snapshot_counters = {}  # {symbol: counter}
        
        # Lock
        self.lock = asyncio.Lock()
        
        self.logger.info(f"Đã khởi tạo OrderbookCollector cho {self.exchange_id}")
    
    def add_update_callback(self, callback: Callable[[str, Dict], None]) -> None:
        """
        Thêm callback để xử lý mỗi khi sổ lệnh cập nhật.
        
        Args:
            callback: Hàm callback(symbol, metrics)
        """
        self._update_callbacks.append(callback)
    
    def add_snapshot_callback(self, callback: Callable[[str, OrderbookSnapshot], None]) -> None:
        """
        Thêm callback để xử lý mỗi khi có snapshot mới.
        
        Args:
            callback: Hàm callback(symbol, snapshot)
        """
        self._snapshot_callbacks.append(callback)
    
    async def _notify_update(self, symbol: str) -> None:
        """
        Thông báo cho các callbacks khi có cập nhật.
        
        Args:
            symbol: Cặp giao dịch
        """
        if symbol not in self.orderbook_managers:
            return
        
        manager = self.orderbook_managers[symbol]
        
        # Tính toán chỉ số
        metrics = manager.calculate_liquidity_metrics()
        
        # Song song hóa gọi callbacks
        tasks = []
        for callback in self._update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Tạo task cho callback bất đồng bộ
                    task = asyncio.create_task(callback(symbol, metrics))
                else:
                    # Gọi callback đồng bộ ngay lập tức
                    callback(symbol, metrics)
                    continue
                    
                tasks.append(task)
            except Exception as e:
                self.logger.error(f"Lỗi khi tạo task cho update callback: {e}")
        
        # Đợi tất cả các callback bất đồng bộ hoàn thành (nếu có)
        if tasks:
            # Sử dụng gather kèm return_exceptions=True để tránh một callback lỗi ảnh hưởng đến các callback khác
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Kiểm tra các lỗi
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Callback thứ {i} gặp lỗi: {result}")
    
    async def _notify_snapshot(self, symbol: str, snapshot: OrderbookSnapshot) -> None:
        """
        Thông báo cho các callbacks khi có snapshot mới.
        
        Args:
            symbol: Cặp giao dịch
            snapshot: Snapshot mới
        """
        # Song song hóa gọi callbacks
        tasks = []
        for callback in self._snapshot_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Tạo task cho callback bất đồng bộ
                    task = asyncio.create_task(callback(symbol, snapshot))
                else:
                    # Gọi callback đồng bộ ngay lập tức
                    callback(symbol, snapshot)
                    continue
                    
                tasks.append(task)
            except Exception as e:
                self.logger.error(f"Lỗi khi tạo task cho snapshot callback: {e}")
        
        # Đợi tất cả các callback bất đồng bộ hoàn thành (nếu có)
        if tasks:
            # Sử dụng gather kèm return_exceptions=True để tránh một callback lỗi ảnh hưởng đến các callback khác
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Kiểm tra các lỗi
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Callback thứ {i} gặp lỗi: {result}")
    
    async def initialize_orderbook(self, symbol: str) -> bool:
        """
        Khởi tạo sổ lệnh cho một cặp giao dịch.
        
        Args:
            symbol: Cặp giao dịch
            
        Returns:
            True nếu khởi tạo thành công, False nếu không
        """
        # Kiểm tra symbol
        if not is_valid_trading_pair(symbol):
            self.logger.error(f"Symbol không hợp lệ: {symbol}")
            return False
            
        async with self.lock:
            if symbol in self.orderbook_managers:
                # Reset nếu đã tồn tại
                self.orderbook_managers[symbol].reset()
            else:
                # Tạo mới nếu chưa có
                self.orderbook_managers[symbol] = OrderbookManager(
                    symbol=symbol,
                    max_depth=self.max_depth,
                    buffer_size=self.buffer_size
                )
            
            # Khởi tạo buffer lưu snapshot
            if symbol not in self._snapshot_buffers:
                self._snapshot_buffers[symbol] = []
                self._snapshot_counters[symbol] = 0
            
            try:
                # Lấy snapshot ban đầu
                orderbook_data = await self.exchange_connector.fetch_order_book(
                    symbol=symbol,
                    limit=self.max_depth
                )
                
                # Tạo snapshot
                snapshot = OrderbookSnapshot.from_ccxt_format(
                    symbol=symbol,
                    orderbook=orderbook_data
                )
                
                # Cập nhật sổ lệnh
                self.orderbook_managers[symbol].update_from_snapshot(snapshot)
                
                # Thông báo snapshot
                await self._notify_snapshot(symbol, snapshot)
                
                # Thông báo cập nhật
                await self._notify_update(symbol)
                
                # Thêm vào buffer lưu trữ
                self._snapshot_buffers[symbol].append(snapshot.to_dict())
                self._snapshot_counters[symbol] += 1
                
                # Kiểm tra xem đã đến ngưỡng lưu file chưa
                if self._snapshot_counters[symbol] >= self.max_file_snapshots:
                    await self._save_snapshot_buffer(symbol)
                
                self.logger.info(f"Đã khởi tạo sổ lệnh cho {symbol}")
                return True
            
            except Exception as e:
                self.logger.error(f"Lỗi khi khởi tạo sổ lệnh cho {symbol}: {e}")
                return False
    
    async def update_from_websocket(self, message: Dict) -> bool:
        """
        Cập nhật sổ lệnh từ dữ liệu websocket.
        
        Args:
            message: Dữ liệu từ websocket
            
        Returns:
            True nếu cập nhật thành công, False nếu không
        """
        # Kiểm tra message
        if not isinstance(message, dict):
            self.logger.error(f"Message không hợp lệ: {type(message)}")
            return False
            
        # Xác định symbol và kiểm tra định dạng
        if 'type' not in message or message['type'] != 'orderbook':
            return False
        
        symbol = message.get('symbol')
        if not symbol:
            self.logger.warning(f"Không tìm thấy symbol trong thông điệp: {message}")
            return False
        
        # Kiểm tra xem có manager cho symbol không
        if symbol not in self.orderbook_managers:
            success = await self.initialize_orderbook(symbol)
            if not success:
                return False
        
        manager = self.orderbook_managers[symbol]
        
        # Xử lý dữ liệu dựa vào định dạng
        if 'bids' in message and 'asks' in message:
            # Đây là một snapshot
            try:
                snapshot = OrderbookSnapshot(
                    symbol=symbol,
                    timestamp=message.get('timestamp', int(time.time() * 1000)),
                    bids=message.get('bids', []),
                    asks=message.get('asks', []),
                    last_update_id=message.get('nonce')
                )
                
                # Cập nhật sổ lệnh
                result = manager.update_from_snapshot(snapshot)
                
                # Thông báo snapshot
                if result:
                    await self._notify_snapshot(symbol, snapshot)
                    
                    # Thông báo cập nhật
                    await self._notify_update(symbol)
                    
                    # Thêm vào buffer lưu trữ
                    async with self.lock:
                        self._snapshot_buffers[symbol].append(snapshot.to_dict())
                        self._snapshot_counters[symbol] += 1
                        
                        # Kiểm tra xem đã đến ngưỡng lưu file chưa
                        if self._snapshot_counters[symbol] >= self.max_file_snapshots:
                            await self._save_snapshot_buffer(symbol)
                
                return result
            except Exception as e:
                self.logger.error(f"Lỗi khi xử lý snapshot từ websocket: {e}")
                return False
            
        else:
            # Đây là một delta (cập nhật gia tăng)
            try:
                result = manager.update_from_delta(message)
                
                # Thông báo cập nhật
                if result:
                    await self._notify_update(symbol)
                
                return result
            except Exception as e:
                self.logger.error(f"Lỗi khi xử lý delta từ websocket: {e}")
                return False
    
    async def _collect_snapshots_task(self, symbol: str) -> None:
        """
        Task thu thập snapshot theo định kỳ.
        
        Args:
            symbol: Cặp giao dịch
        """
        self.logger.info(f"Bắt đầu thu thập snapshots cho {symbol}")
        
        try:
            while self.is_running:
                try:
                    # Lấy snapshot mới
                    orderbook_data = await self.exchange_connector.fetch_order_book(
                        symbol=symbol,
                        limit=self.max_depth
                    )
                    
                    # Tạo snapshot
                    snapshot = OrderbookSnapshot.from_ccxt_format(
                        symbol=symbol,
                        orderbook=orderbook_data
                    )
                    
                    # Cập nhật sổ lệnh
                    if symbol in self.orderbook_managers:
                        self.orderbook_managers[symbol].update_from_snapshot(snapshot)
                        
                        # Thông báo snapshot
                        await self._notify_snapshot(symbol, snapshot)
                        
                        # Thông báo cập nhật
                        await self._notify_update(symbol)
                        
                        # Thêm vào buffer lưu trữ
                        async with self.lock:
                            self._snapshot_buffers[symbol].append(snapshot.to_dict())
                            self._snapshot_counters[symbol] += 1
                            
                            # Kiểm tra xem đã đến ngưỡng lưu file chưa
                            if self._snapshot_counters[symbol] >= self.max_file_snapshots:
                                await self._save_snapshot_buffer(symbol)
                    
                except Exception as e:
                    self.logger.error(f"Lỗi khi thu thập snapshot cho {symbol}: {e}")
                
                # Đợi đến lần tiếp theo
                await asyncio.sleep(self.snapshot_interval)
        
        except asyncio.CancelledError:
            self.logger.info(f"Task thu thập snapshots cho {symbol} đã bị hủy")
            
            # Lưu buffer còn lại khi task bị hủy
            if symbol in self._snapshot_buffers and self._snapshot_buffers[symbol]:
                await self._save_snapshot_buffer(symbol)
                
        except Exception as e:
            self.logger.error(f"Lỗi trong task thu thập snapshots cho {symbol}: {e}")
    
    async def _save_snapshot_buffer(self, symbol: str) -> None:
        """
        Lưu buffer snapshot vào file.
        
        Args:
            symbol: Cặp giao dịch
        """
        if symbol not in self._snapshot_buffers or not self._snapshot_buffers[symbol]:
            return
            
        # Tạo thư mục cho symbol nếu chưa có
        symbol_dir = self.data_dir / symbol.replace('/', '_')
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Lấy buffer hiện tại và reset
        buffer = self._snapshot_buffers[symbol]
        self._snapshot_buffers[symbol] = []
        self._snapshot_counters[symbol] = 0
        
        if not buffer:
            return
            
        # Tạo tên file với thời gian bắt đầu và kết thúc
        start_time = datetime.fromtimestamp(buffer[0]['timestamp'] / 1000).strftime('%Y%m%d_%H%M%S')
        end_time = datetime.fromtimestamp(buffer[-1]['timestamp'] / 1000).strftime('%Y%m%d_%H%M%S')
        filename = f"orderbook_{symbol.replace('/', '_')}_{start_time}_to_{end_time}.parquet"
        file_path = symbol_dir / filename
        
        try:
            # Chuyển đổi thành DataFrame
            df = pd.DataFrame(buffer)
            
            # Chuẩn hóa cột bids và asks từ json sang string
            if 'bids' in df.columns:
                df['bids'] = df['bids'].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
            
            if 'asks' in df.columns:
                df['asks'] = df['asks'].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
            
            # Ghi vào file parquet (nén tốt hơn)
            df.to_parquet(file_path, index=False, compression='snappy')
            
            self.logger.info(f"Đã lưu {len(buffer)} snapshots cho {symbol} vào {file_path}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu snapshots cho {symbol}: {e}")
            
            # Lưu buffer khi có lỗi
            fallback_file = symbol_dir / f"backup_{filename}.json"
            try:
                with open(fallback_file, 'w') as f:
                    json.dump(buffer, f, indent=2)
                self.logger.info(f"Đã lưu backup vào {fallback_file}")
            except Exception as backup_error:
                self.logger.error(f"Không thể lưu backup: {backup_error}")
    
    async def start(self, symbols: List[str]) -> None:
        """
        Bắt đầu thu thập dữ liệu sổ lệnh.
        
        Args:
            symbols: Danh sách cặp giao dịch
        """
        # Kiểm tra danh sách symbols
        if not symbols:
            raise ValueError("Danh sách symbols không được rỗng")
            
        # Kiểm tra từng symbol
        valid_symbols = []
        for symbol in symbols:
            if is_valid_trading_pair(symbol):
                valid_symbols.append(symbol)
            else:
                self.logger.warning(f"Bỏ qua symbol không hợp lệ: {symbol}")
        
        if not valid_symbols:
            raise ValueError("Không có symbol hợp lệ nào để thu thập")
            
        async with self.lock:
            if self.is_running:
                self.logger.warning("OrderbookCollector đã đang chạy")
                return
            
            self.is_running = True
            
            # Khởi tạo sổ lệnh cho mỗi symbol
            for symbol in valid_symbols:
                success = await self.initialize_orderbook(symbol)
                
                if success:
                    # Tạo task thu thập snapshot
                    task = asyncio.create_task(self._collect_snapshots_task(symbol))
                    self.tasks.append(task)
            
            self.logger.info(f"Đã bắt đầu OrderbookCollector cho {len(self.tasks)} symbols")
    
    async def stop(self) -> None:
        """
        Dừng thu thập dữ liệu sổ lệnh.
        """
        async with self.lock:
            if not self.is_running:
                self.logger.warning("OrderbookCollector không chạy")
                return
            
            self.is_running = False
            
            # Hủy tất cả các tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.tasks.clear()
            
            # Lưu tất cả các buffer
            for symbol in list(self._snapshot_buffers.keys()):
                if self._snapshot_buffers[symbol]:
                    await self._save_snapshot_buffer(symbol)
            
            self.logger.info("Đã dừng OrderbookCollector")
    
    def get_orderbook(self, symbol: str) -> Optional[OrderbookSnapshot]:
        """
        Lấy snapshot hiện tại của sổ lệnh.
        
        Args:
            symbol: Cặp giao dịch
            
        Returns:
            Snapshot hiện tại hoặc None nếu không có
        """
        # Kiểm tra symbol
        if not is_valid_trading_pair(symbol):
            self.logger.error(f"Symbol không hợp lệ: {symbol}")
            return None
            
        if symbol not in self.orderbook_managers:
            return None
        
        return self.orderbook_managers[symbol].get_current_snapshot()
    
    def get_liquidity_metrics(self, symbol: str) -> Optional[Dict]:
        """
        Lấy các chỉ số thanh khoản cho một cặp giao dịch.
        
        Args:
            symbol: Cặp giao dịch
            
        Returns:
            Dict với các chỉ số hoặc None nếu không có
        """
        # Kiểm tra symbol
        if not is_valid_trading_pair(symbol):
            self.logger.error(f"Symbol không hợp lệ: {symbol}")
            return None
            
        if symbol not in self.orderbook_managers:
            return None
        
        return self.orderbook_managers[symbol].calculate_liquidity_metrics()
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """
        Lấy các chỉ số thanh khoản cho tất cả các cặp giao dịch.
        
        Returns:
            Dict với key là symbol và value là các chỉ số
        """
        result = {}
        
        for symbol, manager in self.orderbook_managers.items():
            result[symbol] = manager.calculate_liquidity_metrics()
        
        return result
    
    def reset(self, symbol: Optional[str] = None) -> None:
        """
        Reset sổ lệnh.
        
        Args:
            symbol: Cặp giao dịch (None để reset tất cả)
        """
        if symbol:
            # Kiểm tra symbol
            if not is_valid_trading_pair(symbol):
                self.logger.error(f"Symbol không hợp lệ: {symbol}")
                return
                
            if symbol in self.orderbook_managers:
                self.orderbook_managers[symbol].reset()
        else:
            for manager in self.orderbook_managers.values():
                manager.reset()
    
    async def get_historical_snapshots(
        self, 
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[OrderbookSnapshot]:
        """
        Lấy các snapshot lịch sử từ file.
        
        Args:
            symbol: Cặp giao dịch
            start_time: Thời gian bắt đầu
            end_time: Thời gian kết thúc
            
        Returns:
            Danh sách các snapshot
        """
        # Kiểm tra symbol
        if not is_valid_trading_pair(symbol):
            self.logger.error(f"Symbol không hợp lệ: {symbol}")
            return []
            
        if end_time is None:
            end_time = datetime.now()
        
        if start_time is None:
            start_time = end_time - timedelta(days=1)
        
        # Tạo đường dẫn thư mục
        symbol_dir = self.data_dir / symbol.replace('/', '_')
        
        if not symbol_dir.exists():
            self.logger.warning(f"Không tìm thấy thư mục dữ liệu cho {symbol}")
            return []
        
        # Tìm các file trong khoảng thời gian
        all_files = list(symbol_dir.glob(f"orderbook_{symbol.replace('/', '_')}*.parquet")) + \
                   list(symbol_dir.glob(f"orderbook_{symbol.replace('/', '_')}*.json"))
        
        if not all_files:
            self.logger.warning(f"Không tìm thấy file snapshot cho {symbol}")
            return []
        
        # Lọc các file theo thời gian (dựa vào tên file)
        filtered_files = []
        for file_path in all_files:
            # Trích xuất thời gian từ tên file
            try:
                # Tìm tất cả các pattern thời gian trong tên file
                matches = re.findall(r'(\d{8}_\d{6})', file_path.name)
                if matches:
                    # Lấy thời gian đầu tiên và cuối cùng
                    file_start_time = datetime.strptime(matches[0], '%Y%m%d_%H%M%S')
                    file_end_time = datetime.strptime(matches[-1], '%Y%m%d_%H%M%S') if len(matches) > 1 else file_start_time
                    
                    # Kiểm tra xem file có overlap với khoảng thời gian không
                    if (file_start_time <= end_time and file_end_time >= start_time):
                        filtered_files.append(file_path)
            except Exception as e:
                self.logger.warning(f"Không thể trích xuất thời gian từ tên file {file_path.name}: {e}")
                # Thêm vào để xem xét nội dung
                filtered_files.append(file_path)
        
        # Lọc các snapshot theo thời gian
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        snapshots = []
        
        for file_path in filtered_files:
            try:
                if file_path.suffix == '.parquet':
                    # Đọc file parquet
                    df = pd.read_parquet(file_path)
                    
                    # Lọc theo timestamp
                    if 'timestamp' in df.columns:
                        df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
                    
                    # Chuyển đổi cột bids và asks từ string sang list
                    if 'bids' in df.columns and df['bids'].dtype == 'object':
                        df['bids'] = df['bids'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                    
                    if 'asks' in df.columns and df['asks'].dtype == 'object':
                        df['asks'] = df['asks'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                    
                    # Tạo snapshots
                    for _, row in df.iterrows():
                        snapshot = OrderbookSnapshot.from_dict(row.to_dict())
                        snapshots.append(snapshot)
                        
                elif file_path.suffix == '.json':
                    # Đọc file json
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Xử lý dựa vào cấu trúc
                    if isinstance(data, list):
                        # Danh sách các snapshot
                        for item in data:
                            if isinstance(item, dict) and 'timestamp' in item:
                                timestamp = item.get('timestamp', 0)
                                if start_ts <= timestamp <= end_ts:
                                    snapshot = OrderbookSnapshot.from_dict(item)
                                    snapshots.append(snapshot)
                    elif isinstance(data, dict) and 'timestamp' in data:
                        # Một snapshot duy nhất
                        timestamp = data.get('timestamp', 0)
                        if start_ts <= timestamp <= end_ts:
                            snapshot = OrderbookSnapshot.from_dict(data)
                            snapshots.append(snapshot)
                
            except Exception as e:
                self.logger.error(f"Lỗi khi đọc file {file_path}: {e}")
        
        # Sắp xếp theo thời gian
        snapshots.sort(key=lambda x: x.timestamp)
        
        return snapshots


# Factory function
async def create_orderbook_collector(
    exchange_id: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    sandbox: bool = True,
    is_futures: bool = False,
    snapshot_interval: int = 60,
    max_file_snapshots: int = 1000
) -> OrderbookCollector:
    """
    Tạo một instance của OrderbookCollector cho sàn giao dịch cụ thể.
    
    Args:
        exchange_id: ID của sàn giao dịch
        api_key: Khóa API
        api_secret: Mật khẩu API
        sandbox: Sử dụng môi trường testnet
        is_futures: Sử dụng tài khoản futures
        snapshot_interval: Khoảng thời gian giữa các snapshot (giây)
        max_file_snapshots: Số lượng snapshot tối đa trong 1 file
        
    Returns:
        Instance của OrderbookCollector
    """
    # Kiểm tra tham số đầu vào
    if not exchange_id:
        raise ValueError("exchange_id không được rỗng")
        
    if snapshot_interval <= 0:
        raise ValueError("snapshot_interval phải là số dương")
        
    if max_file_snapshots <= 0:
        raise ValueError("max_file_snapshots phải là số dương")
        
    # Tạo connector cho sàn giao dịch
    exchange_connector = None
    
    if exchange_id.lower() == 'binance':
        exchange_connector = BinanceConnector(
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            futures=is_futures
        )
    elif exchange_id.lower() == 'bybit':
        exchange_connector = BybitConnector(
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            category='linear' if is_futures else 'spot'
        )
    else:
        # Sử dụng GenericExchangeConnector cho các sàn khác
        exchange_connector = GenericExchangeConnector(
            exchange_id=exchange_id,
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox
        )
    
    # Khởi tạo connector
    await exchange_connector.initialize()
    
    # Tạo collector
    collector = OrderbookCollector(
        exchange_connector=exchange_connector,
        snapshot_interval=snapshot_interval,
        max_file_snapshots=max_file_snapshots
    )
    
    return collector


class MarketLiquidityMonitor:
    """
    Lớp theo dõi thanh khoản thị trường và phát hiện các xu hướng/biến động.
    """
    
    def __init__(
        self, 
        orderbook_collector: OrderbookCollector,
        alert_threshold: float = 20.0,  # Phần trăm thay đổi để phát cảnh báo
        window_size: int = 10  # Kích thước cửa sổ lịch sử
    ):
        """
        Khởi tạo MarketLiquidityMonitor.
        
        Args:
            orderbook_collector: Bộ thu thập sổ lệnh
            alert_threshold: Ngưỡng phần trăm thay đổi để phát cảnh báo
            window_size: Kích thước cửa sổ lịch sử
        """
        # Kiểm tra tham số đầu vào
        if not isinstance(orderbook_collector, OrderbookCollector):
            raise ValueError("orderbook_collector phải là instance của OrderbookCollector")
            
        if alert_threshold <= 0:
            raise ValueError("alert_threshold phải là số dương")
            
        if window_size <= 1:
            raise ValueError("window_size phải lớn hơn 1")
            
        self.orderbook_collector = orderbook_collector
        self.alert_threshold = alert_threshold
        self.window_size = window_size
        
        # Theo dõi thanh khoản
        self.liquidity_history = {}  # {symbol: deque(metrics)}
        
        # Callbacks
        self._alert_callbacks = []
        
        # Logger
        self.logger = setup_logger("market_liquidity_monitor")
        
        # Đăng ký callback để nhận cập nhật sổ lệnh
        self.orderbook_collector.add_update_callback(self._on_orderbook_update)
        
        self.logger.info(f"Đã khởi tạo MarketLiquidityMonitor")
    
    def add_alert_callback(self, callback: Callable[[str, Dict], None]) -> None:
        """
        Thêm callback để nhận thông báo khi có cảnh báo thanh khoản.
        
        Args:
            callback: Hàm callback(symbol, alert_data)
        """
        self._alert_callbacks.append(callback)
    
    async def _on_orderbook_update(self, symbol: str, metrics: Dict) -> None:
        """
        Xử lý khi có cập nhật sổ lệnh.
        
        Args:
            symbol: Cặp giao dịch
            metrics: Chỉ số thanh khoản
        """
        # Kiểm tra tham số đầu vào
        if not symbol or not isinstance(metrics, dict):
            return
            
        # Tạo history cho symbol nếu chưa có
        if symbol not in self.liquidity_history:
            self.liquidity_history[symbol] = collections.deque(maxlen=self.window_size)
        
        # Thêm vào history
        self.liquidity_history[symbol].append(metrics)
        
        # Phân tích xu hướng nếu đủ dữ liệu
        if len(self.liquidity_history[symbol]) >= 2:
            await self._analyze_trends(symbol)
    
    async def _analyze_trends(self, symbol: str) -> None:
        """
        Phân tích xu hướng thanh khoản và phát hiện biến động.
        
        Args:
            symbol: Cặp giao dịch
        """
        history = self.liquidity_history[symbol]
        
        if len(history) < 2:
            return
        
        # Lấy chỉ số gần nhất và trước đó
        latest = history[-1]
        previous = history[-2]
        
        # Tính toán các thay đổi phần trăm
        try:
            # Kiểm tra các trường bắt buộc
            required_fields = ['mid_price', 'bid_volume_1pct', 'ask_volume_1pct', 'imbalance', 'spread_percentage']
            for field in required_fields:
                if field not in latest or field not in previous:
                    return
                    
            # Tránh chia cho 0
            if previous['mid_price'] == 0 or previous['spread_percentage'] == 0:
                return
                
            price_change_pct = ((latest['mid_price'] - previous['mid_price']) / previous['mid_price']) * 100
            
            bid_vol_change_pct = ((latest['bid_volume_1pct'] - previous['bid_volume_1pct']) / previous['bid_volume_1pct']) * 100 if previous['bid_volume_1pct'] else 0
            
            ask_vol_change_pct = ((latest['ask_volume_1pct'] - previous['ask_volume_1pct']) / previous['ask_volume_1pct']) * 100 if previous['ask_volume_1pct'] else 0
            
            imbalance_change = latest['imbalance'] - previous['imbalance']
            
            spread_change_pct = ((latest['spread_percentage'] - previous['spread_percentage']) / previous['spread_percentage']) * 100 if previous['spread_percentage'] else 0
            
            # Phát hiện các biến động lớn
            alerts = []
            
            if abs(price_change_pct) > self.alert_threshold / 2:
                alerts.append({
                    'type': 'price',
                    'change_pct': price_change_pct,
                    'message': f"Thay đổi giá lớn: {price_change_pct:.2f}%"
                })
            
            if abs(bid_vol_change_pct) > self.alert_threshold:
                alerts.append({
                    'type': 'bid_volume',
                    'change_pct': bid_vol_change_pct,
                    'message': f"Thay đổi khối lượng mua lớn: {bid_vol_change_pct:.2f}%"
                })
            
            if abs(ask_vol_change_pct) > self.alert_threshold:
                alerts.append({
                    'type': 'ask_volume',
                    'change_pct': ask_vol_change_pct,
                    'message': f"Thay đổi khối lượng bán lớn: {ask_vol_change_pct:.2f}%"
                })
            
            if abs(imbalance_change) > 0.3:  # Thay đổi > 0.3 (thang -1 đến 1)
                alerts.append({
                    'type': 'imbalance',
                    'change': imbalance_change,
                    'message': f"Thay đổi mất cân bằng lớn: {imbalance_change:.2f}"
                })
            
            if abs(spread_change_pct) > self.alert_threshold * 2:
                alerts.append({
                    'type': 'spread',
                    'change_pct': spread_change_pct,
                    'message': f"Thay đổi spread lớn: {spread_change_pct:.2f}%"
                })
            
            # Gửi cảnh báo nếu có
            if alerts:
                alert_data = {
                    'symbol': symbol,
                    'timestamp': latest['timestamp'],
                    'datetime': datetime.fromtimestamp(latest['timestamp'] / 1000).isoformat(),
                    'metrics': {
                        'price': latest['mid_price'],
                        'bid_volume': latest['bid_volume_1pct'],
                        'ask_volume': latest['ask_volume_1pct'],
                        'imbalance': latest['imbalance'],
                        'spread': latest['spread_percentage']
                    },
                    'changes': {
                        'price_change_pct': price_change_pct,
                        'bid_vol_change_pct': bid_vol_change_pct,
                        'ask_vol_change_pct': ask_vol_change_pct,
                        'imbalance_change': imbalance_change,
                        'spread_change_pct': spread_change_pct
                    },
                    'alerts': alerts
                }
                
                # Song song hóa gọi callbacks
                tasks = []
                for callback in self._alert_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            # Tạo task cho callback bất đồng bộ
                            task = asyncio.create_task(callback(symbol, alert_data))
                        else:
                            # Gọi callback đồng bộ ngay lập tức
                            callback(symbol, alert_data)
                            continue
                            
                        tasks.append(task)
                    except Exception as e:
                        self.logger.error(f"Lỗi khi tạo task cho alert callback: {e}")
                
                # Đợi tất cả các callback bất đồng bộ hoàn thành (nếu có)
                if tasks:
                    # Sử dụng gather kèm return_exceptions=True để tránh một callback lỗi ảnh hưởng đến các callback khác
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Kiểm tra các lỗi
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            self.logger.error(f"Callback thứ {i} gặp lỗi: {result}")
                
                # Ghi log
                alert_msg = ", ".join([alert['message'] for alert in alerts])
                self.logger.warning(f"Cảnh báo thanh khoản {symbol}: {alert_msg}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi phân tích xu hướng cho {symbol}: {e}")
    
    def get_liquidity_trend(self, symbol: str) -> Dict:
        """
        Lấy xu hướng thanh khoản cho một cặp giao dịch.
        
        Args:
            symbol: Cặp giao dịch
            
        Returns:
            Dict với thông tin xu hướng
        """
        # Kiểm tra symbol
        if not is_valid_trading_pair(symbol):
            return {
                'symbol': symbol,
                'has_data': False,
                'message': "Symbol không hợp lệ"
            }
            
        if symbol not in self.liquidity_history or len(self.liquidity_history[symbol]) < 2:
            return {
                'symbol': symbol,
                'has_data': False,
                'message': "Chưa đủ dữ liệu"
            }
        
        history = list(self.liquidity_history[symbol])
        
        # Tính toán xu hướng
        first_metrics = history[0]
        last_metrics = history[-1]
        
        # Kiểm tra các trường bắt buộc
        required_fields = ['timestamp', 'mid_price', 'bid_volume_1pct', 'ask_volume_1pct', 'imbalance', 'spread_percentage']
        for field in required_fields:
            if field not in first_metrics or field not in last_metrics:
                return {
                    'symbol': symbol,
                    'has_data': False,
                    'message': f"Thiếu trường dữ liệu: {field}"
                }
        
        # Tính toán các thay đổi
        try:
            price_change_pct = ((last_metrics['mid_price'] - first_metrics['mid_price']) / first_metrics['mid_price']) * 100 if first_metrics['mid_price'] else 0
            
            bid_volume_change_pct = ((last_metrics['bid_volume_1pct'] - first_metrics['bid_volume_1pct']) / first_metrics['bid_volume_1pct']) * 100 if first_metrics['bid_volume_1pct'] else 0
            
            ask_volume_change_pct = ((last_metrics['ask_volume_1pct'] - first_metrics['ask_volume_1pct']) / first_metrics['ask_volume_1pct']) * 100 if first_metrics['ask_volume_1pct'] else 0
            
            imbalance_change = last_metrics['imbalance'] - first_metrics['imbalance']
            
            spread_change_pct = ((last_metrics['spread_percentage'] - first_metrics['spread_percentage']) / first_metrics['spread_percentage']) * 100 if first_metrics['spread_percentage'] else 0
            
            return {
                'symbol': symbol,
                'has_data': True,
                'period': {
                    'start_time': datetime.fromtimestamp(first_metrics['timestamp'] / 1000).isoformat(),
                    'end_time': datetime.fromtimestamp(last_metrics['timestamp'] / 1000).isoformat(),
                    'samples': len(history)
                },
                'price': {
                    'first': first_metrics['mid_price'],
                    'last': last_metrics['mid_price'],
                    'change_pct': price_change_pct
                },
                'liquidity': {
                    'first_bid_volume': first_metrics['bid_volume_1pct'],
                    'last_bid_volume': last_metrics['bid_volume_1pct'],
                    'bid_volume_change_pct': bid_volume_change_pct,
                    
                    'first_ask_volume': first_metrics['ask_volume_1pct'],
                    'last_ask_volume': last_metrics['ask_volume_1pct'],
                    'ask_volume_change_pct': ask_volume_change_pct,
                    
                    'first_imbalance': first_metrics['imbalance'],
                    'last_imbalance': last_metrics['imbalance'],
                    'imbalance_change': imbalance_change,
                    
                    'first_spread': first_metrics['spread_percentage'],
                    'last_spread': last_metrics['spread_percentage'],
                    'spread_change_pct': spread_change_pct
                }
            }
        except Exception as e:
            self.logger.error(f"Lỗi khi tính toán xu hướng cho {symbol}: {e}")
            return {
                'symbol': symbol,
                'has_data': False,
                'message': f"Lỗi khi tính toán xu hướng: {str(e)}"
            }
    
    def get_all_trends(self) -> Dict[str, Dict]:
        """
        Lấy xu hướng thanh khoản cho tất cả các cặp giao dịch.
        
        Returns:
            Dict với key là symbol và value là xu hướng
        """
        result = {}
        
        for symbol in self.liquidity_history:
            result[symbol] = self.get_liquidity_trend(symbol)
        
        return result


async def main():
    """
    Hàm chính để chạy orderbook collector.
    """
    # Đọc thông tin cấu hình từ biến môi trường
    exchange_id = get_env('DEFAULT_EXCHANGE', 'binance')
    api_key = get_env(f'{exchange_id.upper()}_API_KEY', '')
    api_secret = get_env(f'{exchange_id.upper()}_API_SECRET', '')
    
    try:
        # Tạo collector
        collector = await create_orderbook_collector(
            exchange_id=exchange_id,
            api_key=api_key,
            api_secret=api_secret,
            sandbox=True,
            snapshot_interval=60,
            max_file_snapshots=1000
        )
        
        # Tạo monitor
        monitor = MarketLiquidityMonitor(
            orderbook_collector=collector,
            alert_threshold=5.0,
            window_size=10
        )
        
        # Callback để in cảnh báo
        async def print_alert(symbol, alert_data):
            print(f"\n=== CẢNH BÁO THANH KHOẢN {symbol} ===")
            for alert in alert_data['alerts']:
                print(f"- {alert['message']}")
            print(f"Giá hiện tại: {alert_data['metrics']['price']}")
            print(f"Mất cân bằng: {alert_data['metrics']['imbalance']:.4f}")
            print(f"Spread: {alert_data['metrics']['spread']:.4f}%")
            print("=================================\n")
        
        # Đăng ký callback
        monitor.add_alert_callback(print_alert)
        
        # Bắt đầu thu thập dữ liệu
        symbols = ['BTC/USDT', 'ETH/USDT']
        
        # Kiểm tra symbols
        if not symbols:
            raise ValueError("Danh sách symbols không được rỗng")
            
        await collector.start(symbols)
        
        print(f"Đang theo dõi thanh khoản cho {', '.join(symbols)}...")
        print("Nhấn Ctrl+C để dừng.")
        
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nĐang dừng...")
    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        # Dừng collector nếu đã khởi tạo
        if 'collector' in locals():
            await collector.stop()
            
            # Đóng kết nối
            await collector.exchange_connector.close()
            
            print("Đã dừng thu thập dữ liệu")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Đã nhận Ctrl+C, thoát...")