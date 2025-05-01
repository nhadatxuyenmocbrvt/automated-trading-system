"""
Mô phỏng thị trường tài chính.
File này cung cấp các lớp để mô phỏng thị trường tài chính dựa trên dữ liệu lịch sử,
bao gồm biến động giá, thanh khoản, và tác động thị trường để phục vụ huấn luyện và backtest.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
import logging
import random
from datetime import datetime, timedelta
import json
import os
import time

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import OrderType, OrderStatus, PositionSide, TimeInForce
from config.system_config import get_system_config

class MarketSimulator:
    """
    Lớp cơ sở để mô phỏng thị trường tài chính.
    Cung cấp các phương thức để mô phỏng giá, thanh khoản, và tác động thị trường.
    """
    
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        data_path: Optional[Union[str, Path]] = None,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        seed: Optional[int] = None,
        volatility_scale: float = 1.0,
        market_impact_scale: float = 0.0001,
        noise_level: float = 0.0001,
        liquidity_profile: str = "normal",
        special_events_prob: float = 0.01,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo mô phỏng thị trường.
        
        Args:
            data: DataFrame chứa dữ liệu thị trường lịch sử
            data_path: Đường dẫn file dữ liệu nếu data không được cung cấp
            symbol: Cặp giao dịch
            timeframe: Khung thời gian
            seed: Seed cho bộ sinh số ngẫu nhiên
            volatility_scale: Hệ số điều chỉnh biến động
            market_impact_scale: Hệ số tác động thị trường (% thay đổi giá trên 1 đơn vị khối lượng)
            noise_level: Mức độ nhiễu thêm vào giá
            liquidity_profile: Hồ sơ thanh khoản ("thin", "normal", "deep")
            special_events_prob: Xác suất xuất hiện các sự kiện đặc biệt
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("market_simulator")
        
        # Random seed để tái tạo được kết quả
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Tải dữ liệu
        self.data = self._load_data(data, data_path)
        
        # Lưu trữ các tham số
        self.symbol = symbol
        self.timeframe = timeframe
        self.volatility_scale = volatility_scale
        self.market_impact_scale = market_impact_scale
        self.noise_level = noise_level
        self.liquidity_profile = liquidity_profile
        self.special_events_prob = special_events_prob
        
        # Cấu hình hệ thống
        self.config = get_system_config()
        
        # Khởi tạo trạng thái mô phỏng
        self.current_idx = 0
        self.current_timestamp = None
        self.current_prices = {}  # {price_type: value}
        self.current_liquidity = {}  # {side: value}
        self.current_orderbook = None
        self.market_impact_history = []
        
        # Tính toán thông số thị trường
        self._calculate_market_parameters()
        
        # Thêm cột biến động thêm nếu cần
        if "volatility" not in self.data.columns and "close" in self.data.columns:
            self.data["volatility"] = self.data["close"].pct_change().rolling(window=20).std().fillna(0.01)
        
        # Tính toán mô hình thanh khoản dựa trên hồ sơ
        self._setup_liquidity_profile()
        
        self.logger.info(f"Đã khởi tạo MarketSimulator cho {self.symbol} - {self.timeframe}")
    
    def _load_data(
        self, 
        data: Optional[pd.DataFrame] = None, 
        data_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Tải dữ liệu từ DataFrame hoặc file.
        
        Args:
            data: DataFrame chứa dữ liệu
            data_path: Đường dẫn file dữ liệu
            
        Returns:
            DataFrame dữ liệu
        """
        if data is not None:
            return data.copy()
        elif data_path is not None:
            # Tải dữ liệu từ file
            path = Path(data_path)
            if not path.exists():
                raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
            
            if path.suffix == '.csv':
                return pd.read_csv(path)
            elif path.suffix == '.parquet':
                return pd.read_parquet(path)
            elif path.suffix == '.json':
                return pd.read_json(path)
            else:
                raise ValueError(f"Định dạng file không được hỗ trợ: {path.suffix}")
        else:
            self.logger.warning("Không có dữ liệu được cung cấp, sẽ tạo dữ liệu ngẫu nhiên")
            # Tạo dữ liệu ngẫu nhiên cho kiểm thử
            return self._generate_random_data()
    
    def _generate_random_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Tạo dữ liệu ngẫu nhiên cho kiểm thử.
        
        Args:
            n_samples: Số lượng mẫu cần tạo
            
        Returns:
            DataFrame dữ liệu ngẫu nhiên
        """
        # Tạo giá ngẫu nhiên với random walk
        price = 10000.0  # Giá ban đầu
        prices = [price]
        
        for _ in range(n_samples - 1):
            # Random walk với bias nhẹ
            change = np.random.normal(0, 1) * price * 0.01
            price += change
            prices.append(max(price, 100))  # Đảm bảo giá không âm hoặc quá nhỏ
        
        # Tạo OHLCV
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(n_samples-1, -1, -1)]
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'close': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'volume': [abs(np.random.normal(100, 30)) for _ in range(n_samples)]
        })
        
        # Điều chỉnh để đảm bảo high >= open >= low, close <= high, close >= low
        for i in range(n_samples):
            df.loc[i, 'high'] = max(df.loc[i, 'high'], df.loc[i, 'open'], df.loc[i, 'close'])
            df.loc[i, 'low'] = min(df.loc[i, 'low'], df.loc[i, 'open'], df.loc[i, 'close'])
        
        return df
    
    def _calculate_market_parameters(self) -> None:
        """
        Tính toán các thông số thị trường từ dữ liệu lịch sử.
        """
        if self.data is None or len(self.data) == 0:
            self.logger.warning("Không có dữ liệu để tính toán thông số thị trường")
            return
        
        # Tính biến động trung bình
        if 'close' in self.data.columns:
            self.avg_volatility = self.data['close'].pct_change().std()
        else:
            self.avg_volatility = 0.01  # Giá trị mặc định
        
        # Tính khối lượng trung bình
        if 'volume' in self.data.columns:
            self.avg_volume = self.data['volume'].mean()
        else:
            self.avg_volume = 100.0  # Giá trị mặc định
        
        # Tính spread trung bình (giả sử là 0.1% nếu không có dữ liệu)
        if all(col in self.data.columns for col in ['bid', 'ask']):
            spreads = (self.data['ask'] - self.data['bid']) / self.data['ask']
            self.avg_spread = spreads.mean()
        else:
            self.avg_spread = 0.001  # Giá trị mặc định
    
    def _setup_liquidity_profile(self) -> None:
        """
        Thiết lập mô hình thanh khoản dựa trên hồ sơ đã chọn.
        """
        # Các thông số thanh khoản dựa trên hồ sơ
        liquidity_profiles = {
            "thin": {
                "depth_factor": 0.5,
                "spread_factor": 2.0,
                "recovery_rate": 0.5,
                "volatility_impact": 2.0
            },
            "normal": {
                "depth_factor": 1.0,
                "spread_factor": 1.0,
                "recovery_rate": 1.0,
                "volatility_impact": 1.0
            },
            "deep": {
                "depth_factor": 2.0,
                "spread_factor": 0.5,
                "recovery_rate": 1.5,
                "volatility_impact": 0.5
            }
        }
        
        # Lấy hồ sơ thanh khoản
        profile = liquidity_profiles.get(self.liquidity_profile, liquidity_profiles["normal"])
        
        # Lưu thông số
        self.depth_factor = profile["depth_factor"]
        self.spread_factor = profile["spread_factor"]
        self.recovery_rate = profile["recovery_rate"]
        self.volatility_impact = profile["volatility_impact"]
        
        # Tính thanh khoản ban đầu
        self.base_liquidity = self.avg_volume * self.depth_factor
        self.base_spread = self.avg_spread * self.spread_factor
    
    def reset(self, idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Đặt lại mô phỏng thị trường về một điểm cụ thể.
        
        Args:
            idx: Chỉ số trong dữ liệu để đặt lại (None để chọn ngẫu nhiên)
            
        Returns:
            Dict thông tin thị trường hiện tại
        """
        if self.data is None or len(self.data) == 0:
            self.logger.error("Không có dữ liệu để đặt lại mô phỏng")
            return {}
        
        # Xác định chỉ số hiện tại
        if idx is None:
            self.current_idx = random.randint(0, len(self.data) - 1)
        else:
            self.current_idx = max(0, min(idx, len(self.data) - 1))
        
        # Lấy dữ liệu tại chỉ số hiện tại
        self.current_row = self.data.iloc[self.current_idx]
        
        # Lấy timestamp nếu có
        if 'timestamp' in self.current_row:
            self.current_timestamp = self.current_row['timestamp']
        else:
            self.current_timestamp = datetime.now()
        
        # Khởi tạo giá hiện tại
        self.current_prices = {
            'open': float(self.current_row.get('open', 0)),
            'high': float(self.current_row.get('high', 0)),
            'low': float(self.current_row.get('low', 0)),
            'close': float(self.current_row.get('close', 0)),
            'mid': float(self.current_row.get('close', 0)),
            'bid': float(self.current_row.get('bid', self.current_row.get('close', 0) * (1 - self.base_spread/2))),
            'ask': float(self.current_row.get('ask', self.current_row.get('close', 0) * (1 + self.base_spread/2)))
        }
        
        # Khởi tạo thanh khoản hiện tại
        current_volatility = self.current_row.get('volatility', self.avg_volatility)
        volatility_factor = 1.0 / (1.0 + current_volatility * self.volatility_impact)
        
        self.current_liquidity = {
            'bid': self.base_liquidity * volatility_factor,
            'ask': self.base_liquidity * volatility_factor
        }
        
        # Tạo orderbook ban đầu
        self._generate_orderbook()
        
        # Đặt lại lịch sử tác động thị trường
        self.market_impact_history = []
        
        # Trả về thông tin thị trường hiện tại
        return self.get_market_state()
    
    def step(self) -> Dict[str, Any]:
        """
        Tiến thị trường đến điểm tiếp theo trong dữ liệu.
        
        Returns:
            Dict thông tin thị trường mới
        """
        if self.data is None or len(self.data) == 0:
            self.logger.error("Không có dữ liệu để tiến thị trường")
            return {}
        
        # Di chuyển đến bước tiếp theo
        self.current_idx = min(self.current_idx + 1, len(self.data) - 1)
        
        # Lấy dữ liệu mới
        self.current_row = self.data.iloc[self.current_idx]
        
        # Cập nhật timestamp
        if 'timestamp' in self.current_row:
            self.current_timestamp = self.current_row['timestamp']
        else:
            self.current_timestamp = datetime.now() + timedelta(hours=self.current_idx)
        
        # Cập nhật giá hiện tại
        self.current_prices = {
            'open': float(self.current_row.get('open', 0)),
            'high': float(self.current_row.get('high', 0)),
            'low': float(self.current_row.get('low', 0)),
            'close': float(self.current_row.get('close', 0)),
            'mid': float(self.current_row.get('close', 0)),
            'bid': float(self.current_row.get('bid', self.current_row.get('close', 0) * (1 - self.base_spread/2))),
            'ask': float(self.current_row.get('ask', self.current_row.get('close', 0) * (1 + self.base_spread/2)))
        }
        
        # Thêm nhiễu vào giá nếu cần
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level * self.current_prices['close'])
            self.current_prices['mid'] += noise
            self.current_prices['bid'] += noise
            self.current_prices['ask'] += noise
        
        # Cập nhật thanh khoản
        current_volatility = self.current_row.get('volatility', self.avg_volatility)
        volatility_factor = 1.0 / (1.0 + current_volatility * self.volatility_impact)
        
        # Phục hồi thanh khoản theo thời gian
        for side in ['bid', 'ask']:
            self.current_liquidity[side] = (
                self.current_liquidity[side] * (1 - self.recovery_rate) + 
                self.base_liquidity * volatility_factor * self.recovery_rate
            )
        
        # Tạo sự kiện đặc biệt nếu cần
        if random.random() < self.special_events_prob:
            self._generate_special_event()
        
        # Tạo orderbook mới
        self._generate_orderbook()
        
        # Trả về thông tin thị trường hiện tại
        return self.get_market_state()
    
    def _generate_orderbook(self) -> None:
        """
        Tạo orderbook giả định dựa trên giá và thanh khoản hiện tại.
        """
        # Lấy giá hiện tại
        mid_price = self.current_prices['mid']
        bid_price = self.current_prices['bid']
        ask_price = self.current_prices['ask']
        
        # Số lượng cấp giá
        levels = 20
        
        # Tạo orderbook
        bids = []
        asks = []
        
        # Tạo các cấp giá bid
        total_bid_volume = self.current_liquidity['bid']
        bid_volume_per_level = total_bid_volume / sum(i**(-0.8) for i in range(1, levels+1))
        
        current_bid = bid_price
        price_step = mid_price * 0.0005  # 0.05% bước giá
        
        for i in range(1, levels+1):
            volume = bid_volume_per_level * (i**(-0.8))  # Phân phối theo hàm mũ
            bids.append([current_bid, volume])
            current_bid -= price_step * (1 + 0.1 * random.random())  # Thêm chút nhiễu
        
        # Tạo các cấp giá ask
        total_ask_volume = self.current_liquidity['ask']
        ask_volume_per_level = total_ask_volume / sum(i**(-0.8) for i in range(1, levels+1))
        
        current_ask = ask_price
        
        for i in range(1, levels+1):
            volume = ask_volume_per_level * (i**(-0.8))  # Phân phối theo hàm mũ
            asks.append([current_ask, volume])
            current_ask += price_step * (1 + 0.1 * random.random())  # Thêm chút nhiễu
        
        # Tạo orderbook
        self.current_orderbook = {
            'timestamp': self.current_timestamp,
            'mid': mid_price,
            'bids': bids,
            'asks': asks,
            'bid_volumes': sum(b[1] for b in bids),
            'ask_volumes': sum(a[1] for a in asks)
        }
    
    def _generate_special_event(self) -> None:
        """
        Tạo các sự kiện thị trường đặc biệt (flash crash, pump, liquidity shock, etc.).
        """
        # Các loại sự kiện
        event_types = [
            'flash_crash',
            'sudden_pump',
            'liquidity_shock',
            'high_volatility',
            'price_consolidation'
        ]
        
        # Chọn ngẫu nhiên loại sự kiện
        event_type = random.choice(event_types)
        
        # Xử lý từng loại sự kiện
        if event_type == 'flash_crash':
            # Giảm giá đột ngột 5-10%
            drop_factor = random.uniform(0.05, 0.1)
            for price_type in ['mid', 'bid', 'ask']:
                self.current_prices[price_type] *= (1 - drop_factor)
            
            # Giảm thanh khoản bên mua
            self.current_liquidity['bid'] *= 0.3
            
            self.logger.info(f"Sự kiện đặc biệt: Flash crash, giá giảm {drop_factor*100:.1f}%")
            
        elif event_type == 'sudden_pump':
            # Tăng giá đột ngột 5-10%
            pump_factor = random.uniform(0.05, 0.1)
            for price_type in ['mid', 'bid', 'ask']:
                self.current_prices[price_type] *= (1 + pump_factor)
            
            # Giảm thanh khoản bên bán
            self.current_liquidity['ask'] *= 0.3
            
            self.logger.info(f"Sự kiện đặc biệt: Sudden pump, giá tăng {pump_factor*100:.1f}%")
            
        elif event_type == 'liquidity_shock':
            # Giảm thanh khoản cả hai bên
            shock_factor = random.uniform(0.2, 0.5)
            self.current_liquidity['bid'] *= shock_factor
            self.current_liquidity['ask'] *= shock_factor
            
            # Tăng spread
            spread_increase = random.uniform(0.005, 0.02)
            mid_price = self.current_prices['mid']
            self.current_prices['bid'] = mid_price * (1 - spread_increase)
            self.current_prices['ask'] = mid_price * (1 + spread_increase)
            
            self.logger.info(f"Sự kiện đặc biệt: Liquidity shock, thanh khoản giảm {(1-shock_factor)*100:.1f}%")
            
        elif event_type == 'high_volatility':
            # Tăng biến động
            vol_factor = random.uniform(0.02, 0.05)
            mid_price = self.current_prices['mid']
            
            # Dao động giá ngẫu nhiên
            price_change = mid_price * vol_factor * random.uniform(-1, 1)
            for price_type in ['mid', 'bid', 'ask']:
                self.current_prices[price_type] += price_change
            
            self.logger.info(f"Sự kiện đặc biệt: High volatility, biến động ±{vol_factor*100:.1f}%")
            
        elif event_type == 'price_consolidation':
            # Giảm biến động, thu hẹp spread
            spread_decrease = 0.5  # Giảm 50% spread
            mid_price = self.current_prices['mid']
            current_spread = self.current_prices['ask'] - self.current_prices['bid']
            new_spread = current_spread * spread_decrease
            
            self.current_prices['bid'] = mid_price - new_spread/2
            self.current_prices['ask'] = mid_price + new_spread/2
            
            # Tăng thanh khoản
            self.current_liquidity['bid'] *= 1.5
            self.current_liquidity['ask'] *= 1.5
            
            self.logger.info(f"Sự kiện đặc biệt: Price consolidation, spread giảm {(1-spread_decrease)*100:.1f}%")
    
    def simulate_market_impact(
        self, 
        order_volume: float, 
        side: str, 
        order_type: str = OrderType.MARKET.value,
        immediate_impact: bool = True
    ) -> Dict[str, Any]:
        """
        Mô phỏng tác động thị trường từ một lệnh.
        
        Args:
            order_volume: Khối lượng lệnh
            side: Phía lệnh ('buy' hoặc 'sell')
            order_type: Loại lệnh
            immediate_impact: Cập nhật giá ngay lập tức hay không
            
        Returns:
            Dict chứa thông tin tác động thị trường
        """
        if order_volume <= 0:
            return {'price_impact': 0, 'liquidity_impact': 0}
        
        # Chuẩn hóa side
        side = side.lower()
        
        # Tính tác động giá theo công thức: impact = volume * market_impact_scale * sqrt(volume/liquidity)
        relative_volume = order_volume / self.current_liquidity.get(side, self.base_liquidity)
        price_impact = order_volume * self.market_impact_scale * (relative_volume ** 0.5)
        
        # Điều chỉnh dấu tùy thuộc vào phía
        if side == 'sell':
            price_impact = -price_impact
        
        # Tính tác động thanh khoản
        liquidity_impact = min(relative_volume, 0.9)  # Tối đa lấy đi 90% thanh khoản
        
        # Tạo bản ghi tác động
        impact_record = {
            'timestamp': self.current_timestamp,
            'side': side,
            'volume': order_volume,
            'order_type': order_type,
            'price_impact': price_impact,
            'liquidity_impact': liquidity_impact,
            'recovery_steps': int(10 * liquidity_impact / self.recovery_rate)  # Số bước để phục hồi
        }
        
        # Thêm vào lịch sử
        self.market_impact_history.append(impact_record)
        
        # Cập nhật giá và thanh khoản ngay lập tức nếu cần
        if immediate_impact:
            # Cập nhật giá
            self.current_prices['mid'] *= (1 + price_impact)
            self.current_prices['bid'] *= (1 + price_impact)
            self.current_prices['ask'] *= (1 + price_impact)
            
            # Cập nhật thanh khoản
            opposite_side = 'ask' if side == 'bid' else 'bid'
            self.current_liquidity[side] *= (1 - liquidity_impact)
            self.current_liquidity[opposite_side] *= (1 - liquidity_impact * 0.5)  # Tác động phía đối diện nhỏ hơn
            
            # Tạo lại orderbook
            self._generate_orderbook()
        
        return {
            'price_impact': price_impact,
            'liquidity_impact': liquidity_impact,
            'new_mid_price': self.current_prices['mid'],
            'new_bid_price': self.current_prices['bid'],
            'new_ask_price': self.current_prices['ask']
        }
    
    def get_execution_price(
        self, 
        volume: float, 
        side: str, 
        order_type: str = OrderType.MARKET.value,
        limit_price: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Tính giá thực thi cho một lệnh dựa trên khối lượng và thanh khoản.
        
        Args:
            volume: Khối lượng lệnh
            side: Phía lệnh ('buy' hoặc 'sell')
            order_type: Loại lệnh
            limit_price: Giá limit nếu đó là lệnh limit
            
        Returns:
            Tuple (giá thực thi, khối lượng thực thi)
        """
        if volume <= 0:
            return 0.0, 0.0
        
        # Chuẩn hóa side
        side = side.lower()
        
        # Lấy orderbook hiện tại
        if self.current_orderbook is None:
            self._generate_orderbook()
        
        # Quyết định phía của orderbook
        book_side = 'asks' if side == 'buy' else 'bids'
        price_list = self.current_orderbook[book_side]
        
        # Tính giá thực thi trung bình dựa trên khối lượng
        executed_volume = 0
        weighted_price_sum = 0
        
        for price, level_volume in price_list:
            # Nếu là lệnh limit, kiểm tra giá
            if order_type == OrderType.LIMIT.value:
                if (side == 'buy' and price > limit_price) or (side == 'sell' and price < limit_price):
                    break
            
            # Tính khối lượng thực thi ở cấp giá này
            executable = min(level_volume, volume - executed_volume)
            executed_volume += executable
            weighted_price_sum += price * executable
            
            # Nếu đã thực thi đủ khối lượng, dừng lại
            if executed_volume >= volume:
                break
        
        # Tính giá thực thi trung bình
        if executed_volume > 0:
            avg_execution_price = weighted_price_sum / executed_volume
        else:
            # Nếu không thực thi được, trả về giá thị trường
            avg_execution_price = self.current_prices['ask'] if side == 'buy' else self.current_prices['bid']
            executed_volume = 0
        
        return avg_execution_price, executed_volume
    
    def get_market_state(self) -> Dict[str, Any]:
        """
        Lấy trạng thái thị trường hiện tại.
        
        Returns:
            Dict chứa thông tin thị trường
        """
        # Tính giá OHLC và thanh khoản
        state = {
            'timestamp': self.current_timestamp,
            'prices': self.current_prices.copy(),
            'liquidity': self.current_liquidity.copy(),
            'volatility': self.current_row.get('volatility', self.avg_volatility),
            'volume': self.current_row.get('volume', self.avg_volume),
            'orderbook': self.current_orderbook.copy() if self.current_orderbook else None,
            'market_impact_history': self.market_impact_history.copy()
        }
        
        return state
    
    def get_price(self, price_type: str = 'close') -> float:
        """
        Lấy giá hiện tại theo loại.
        
        Args:
            price_type: Loại giá ('open', 'high', 'low', 'close', 'bid', 'ask', 'mid')
            
        Returns:
            Giá hiện tại
        """
        return self.current_prices.get(price_type, self.current_prices.get('close', 0))
    
    def save_state(self, path: Union[str, Path]) -> None:
        """
        Lưu trạng thái mô phỏng thị trường vào file.
        
        Args:
            path: Đường dẫn file
        """
        state = {
            'current_idx': self.current_idx,
            'current_timestamp': self.current_timestamp.isoformat() if hasattr(self.current_timestamp, 'isoformat') else str(self.current_timestamp),
            'current_prices': self.current_prices,
            'current_liquidity': self.current_liquidity,
            'market_impact_history': self.market_impact_history
        }
        
        # Lưu vào file
        with open(path, 'w') as f:
            json.dump(state, f, indent=4)
    
    def load_state(self, path: Union[str, Path]) -> bool:
        """
        Tải trạng thái mô phỏng thị trường từ file.
        
        Args:
            path: Đường dẫn file
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            # Khôi phục trạng thái
            self.current_idx = state.get('current_idx', 0)
            
            # Khôi phục timestamp
            timestamp_str = state.get('current_timestamp')
            try:
                self.current_timestamp = datetime.fromisoformat(timestamp_str)
            except:
                self.current_timestamp = timestamp_str
            
            self.current_prices = state.get('current_prices', {})
            self.current_liquidity = state.get('current_liquidity', {})
            self.market_impact_history = state.get('market_impact_history', [])
            
            # Tạo lại orderbook
            self._generate_orderbook()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải trạng thái: {e}")
            return False


class HistoricalMarketSimulator(MarketSimulator):
    """
    Lớp mô phỏng thị trường dựa trên dữ liệu lịch sử, không thêm nhiễu hay tác động.
    Sử dụng để backtesting với dữ liệu chính xác.
    """
    
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        data_path: Optional[Union[str, Path]] = None,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo mô phỏng thị trường lịch sử.
        
        Args:
            data: DataFrame chứa dữ liệu thị trường lịch sử
            data_path: Đường dẫn file dữ liệu nếu data không được cung cấp
            symbol: Cặp giao dịch
            timeframe: Khung thời gian
            logger: Logger tùy chỉnh
        """
        # Gọi constructor của lớp cha với các tham số không gây nhiễu
        super().__init__(
            data=data,
            data_path=data_path,
            symbol=symbol,
            timeframe=timeframe,
            volatility_scale=1.0,
            market_impact_scale=0.0,  # Không có tác động thị trường
            noise_level=0.0,  # Không có nhiễu
            special_events_prob=0.0,  # Không có sự kiện đặc biệt
            logger=logger
        )
        
        self.logger.info(f"Đã khởi tạo HistoricalMarketSimulator cho {self.symbol} - {self.timeframe}")
    
    def simulate_market_impact(self, order_volume: float, side: str, order_type: str = OrderType.MARKET.value, 
                            immediate_impact: bool = True) -> Dict[str, Any]:
        """
        Mô phỏng tác động thị trường từ một lệnh (không có tác động trong mô phỏng lịch sử).
        
        Args:
            order_volume: Khối lượng lệnh
            side: Phía lệnh ('buy' hoặc 'sell')
            order_type: Loại lệnh
            immediate_impact: Cập nhật giá ngay lập tức hay không
            
        Returns:
            Dict chứa thông tin tác động thị trường
        """
        # Không có tác động thị trường trong mô phỏng lịch sử
        return {
            'price_impact': 0.0,
            'liquidity_impact': 0.0,
            'new_mid_price': self.current_prices['mid'],
            'new_bid_price': self.current_prices['bid'],
            'new_ask_price': self.current_prices['ask']
        }
    
    def get_execution_price(self, volume: float, side: str, order_type: str = OrderType.MARKET.value,
                          limit_price: Optional[float] = None) -> Tuple[float, float]:
        """
        Tính giá thực thi cho một lệnh (sử dụng giá thị trường đơn giản trong mô phỏng lịch sử).
        
        Args:
            volume: Khối lượng lệnh
            side: Phía lệnh ('buy' hoặc 'sell')
            order_type: Loại lệnh
            limit_price: Giá limit nếu đó là lệnh limit
            
        Returns:
            Tuple (giá thực thi, khối lượng thực thi)
        """
        # Xác định giá thực thi dựa trên loại lệnh và phía
        if order_type == OrderType.MARKET.value:
            # Lệnh thị trường sử dụng giá ask/bid
            if side.lower() == 'buy':
                price = self.current_prices['ask']
            else:
                price = self.current_prices['bid']
            
            return price, volume
            
        elif order_type == OrderType.LIMIT.value and limit_price is not None:
            # Lệnh limit kiểm tra giá
            if side.lower() == 'buy':
                if limit_price >= self.current_prices['ask']:
                    return self.current_prices['ask'], volume
                else:
                    return 0.0, 0.0
            else:
                if limit_price <= self.current_prices['bid']:
                    return self.current_prices['bid'], volume
                else:
                    return 0.0, 0.0
        
        # Mặc định không thực thi
        return 0.0, 0.0


class RealisticMarketSimulator(MarketSimulator):
    """
    Lớp mô phỏng thị trường thực tế với độ trễ, trượt giá, và tỷ lệ từ chối lệnh.
    Thích hợp cho việc đánh giá chiến lược trong điều kiện thị trường thực tế.
    """
    
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        data_path: Optional[Union[str, Path]] = None,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        volatility_scale: float = 1.0,
        market_impact_scale: float = 0.0002,
        noise_level: float = 0.0002,
        liquidity_profile: str = "normal",
        execution_delay: float = 0.5,  # Độ trễ thực thi (đơn vị là bước)
        order_rejection_prob: float = 0.05,  # Xác suất từ chối lệnh
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo mô phỏng thị trường thực tế.
        
        Args:
            data: DataFrame chứa dữ liệu thị trường lịch sử
            data_path: Đường dẫn file dữ liệu nếu data không được cung cấp
            symbol: Cặp giao dịch
            timeframe: Khung thời gian
            volatility_scale: Hệ số điều chỉnh biến động
            market_impact_scale: Hệ số tác động thị trường
            noise_level: Mức độ nhiễu thêm vào giá
            liquidity_profile: Hồ sơ thanh khoản
            execution_delay: Độ trễ thực thi (đơn vị là bước)
            order_rejection_prob: Xác suất từ chối lệnh
            logger: Logger tùy chỉnh
        """
        # Gọi constructor của lớp cha
        super().__init__(
            data=data,
            data_path=data_path,
            symbol=symbol,
            timeframe=timeframe,
            volatility_scale=volatility_scale,
            market_impact_scale=market_impact_scale,
            noise_level=noise_level,
            liquidity_profile=liquidity_profile,
            special_events_prob=0.02,  # Tăng xác suất sự kiện đặc biệt
            logger=logger
        )
        
        # Thiết lập thông số bổ sung
        self.execution_delay = execution_delay
        self.order_rejection_prob = order_rejection_prob
        
        # Hàng đợi lệnh
        self.pending_orders = []
        
        self.logger.info(f"Đã khởi tạo RealisticMarketSimulator cho {self.symbol} - {self.timeframe}")
    
    def step(self) -> Dict[str, Any]:
        """
        Tiến thị trường đến điểm tiếp theo và xử lý các lệnh đang chờ.
        
        Returns:
            Dict thông tin thị trường mới
        """
        # Gọi hàm step của lớp cha
        market_state = super().step()
        
        # Xử lý các lệnh đang chờ
        self._process_pending_orders()
        
        return market_state
    
    def submit_order(
        self, 
        volume: float, 
        side: str, 
        order_type: str = OrderType.MARKET.value,
        limit_price: Optional[float] = None,
        time_in_force: str = TimeInForce.GTC.value,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Gửi lệnh vào hàng đợi.
        
        Args:
            volume: Khối lượng lệnh
            side: Phía lệnh ('buy' hoặc 'sell')
            order_type: Loại lệnh
            limit_price: Giá limit nếu đó là lệnh limit
            time_in_force: Hiệu lực thời gian
            callback: Hàm callback khi lệnh được thực thi
            
        Returns:
            Dict thông tin lệnh
        """
        # Tạo ID lệnh
        order_id = f"{side}_{order_type}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        # Tạo thông tin lệnh
        order = {
            'id': order_id,
            'timestamp': self.current_timestamp,
            'side': side.lower(),
            'volume': volume,
            'order_type': order_type,
            'limit_price': limit_price,
            'time_in_force': time_in_force,
            'status': OrderStatus.PENDING.value,
            'execution_time': self.current_idx + max(1, int(self.execution_delay * random.uniform(0.8, 1.2))),
            'executed_price': None,
            'executed_volume': 0.0,
            'callback': callback
        }
        
        # Thêm vào hàng đợi
        self.pending_orders.append(order)
        
        self.logger.debug(f"Đã gửi lệnh {order_id}: {side} {volume} {order_type}")
        
        return {
            'order_id': order_id,
            'status': OrderStatus.PENDING.value,
            'message': f"Lệnh đã được gửi và sẽ thực thi sau khoảng {order['execution_time'] - self.current_idx} bước"
        }
    
    def _process_pending_orders(self) -> None:
        """
        Xử lý các lệnh đang chờ trong hàng đợi.
        """
        if not self.pending_orders:
            return
        
        # Danh sách lệnh đã xử lý để xóa
        processed_orders = []
        
        for order in self.pending_orders:
            # Kiểm tra xem đã đến thời điểm thực thi chưa
            if self.current_idx >= order['execution_time']:
                # Xác định xem lệnh có bị từ chối không
                if random.random() < self.order_rejection_prob:
                    # Từ chối lệnh
                    order['status'] = OrderStatus.REJECTED.value
                    self.logger.debug(f"Lệnh {order['id']} bị từ chối")
                    
                    # Gọi callback nếu có
                    if order['callback']:
                        order['callback'](order)
                    
                    processed_orders.append(order)
                    continue
                
                # Thực thi lệnh
                executed_price, executed_volume = self.get_execution_price(
                    volume=order['volume'],
                    side=order['side'],
                    order_type=order['order_type'],
                    limit_price=order['limit_price']
                )
                
                # Nếu thực thi được
                if executed_volume > 0:
                    # Mô phỏng tác động thị trường
                    self.simulate_market_impact(executed_volume, order['side'], order['order_type'])
                    
                    # Cập nhật thông tin lệnh
                    order['status'] = OrderStatus.FILLED.value if executed_volume >= order['volume'] else OrderStatus.PARTIALLY_FILLED.value
                    order['executed_price'] = executed_price
                    order['executed_volume'] = executed_volume
                    
                    self.logger.debug(f"Lệnh {order['id']} thực thi: {executed_volume} @ {executed_price}")
                    
                    # Gọi callback nếu có
                    if order['callback']:
                        order['callback'](order)
                    
                    # Thêm vào danh sách đã xử lý
                    if order['status'] == OrderStatus.FILLED.value or order['time_in_force'] != TimeInForce.GTC.value:
                        processed_orders.append(order)
                
                # Nếu không thực thi được và lệnh có thời hạn
                elif order['time_in_force'] == TimeInForce.IOC.value or order['time_in_force'] == TimeInForce.FOK.value:
                    order['status'] = OrderStatus.EXPIRED.value
                    self.logger.debug(f"Lệnh {order['id']} hết hạn")
                    
                    # Gọi callback nếu có
                    if order['callback']:
                        order['callback'](order)
                    
                    processed_orders.append(order)
        
        # Xóa các lệnh đã xử lý khỏi hàng đợi
        for order in processed_orders:
            self.pending_orders.remove(order)
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Hủy lệnh đang chờ.
        
        Args:
            order_id: ID lệnh cần hủy
            
        Returns:
            Dict kết quả hủy lệnh
        """
        for i, order in enumerate(self.pending_orders):
            if order['id'] == order_id:
                # Cập nhật trạng thái lệnh
                order['status'] = OrderStatus.CANCELED.value
                
                # Gọi callback nếu có
                if order['callback']:
                    order['callback'](order)
                
                # Xóa khỏi hàng đợi
                self.pending_orders.pop(i)
                
                self.logger.debug(f"Đã hủy lệnh {order_id}")
                
                return {
                    'order_id': order_id,
                    'status': OrderStatus.CANCELED.value,
                    'message': "Lệnh đã được hủy thành công"
                }
        
        # Nếu không tìm thấy lệnh
        return {
            'order_id': order_id,
            'status': 'error',
            'message': "Không tìm thấy lệnh"
        }
    
    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các lệnh đang chờ.
        
        Returns:
            Danh sách các lệnh đang chờ
        """
        # Lọc ra thông tin cần thiết, bỏ callback
        return [{k: v for k, v in order.items() if k != 'callback'} for order in self.pending_orders]