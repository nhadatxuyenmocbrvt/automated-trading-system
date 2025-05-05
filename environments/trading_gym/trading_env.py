"""
Môi trường giao dịch chính.
File này định nghĩa lớp TradingEnv dựa trên gym.Env, tích hợp với dữ liệu thị trường
và các module khác để cung cấp môi trường giao dịch hoàn chỉnh.
"""

import os
import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import PositionSide, OrderType, OrderStatus, RewardFunction
from environments.base_environment import BaseEnvironment
from environments.trading_gym.observation_space import ObservationSpace
from environments.trading_gym.action_space import ActionSpace

# Import các hàm phần thưởng khi có sẵn
try:
    from environments.reward_functions.profit_reward import calculate_profit_reward
    from environments.reward_functions.risk_adjusted_reward import calculate_risk_adjusted_reward
    from environments.reward_functions.custom_reward import calculate_custom_reward
    REWARD_FUNCTIONS_AVAILABLE = True
except ImportError:
    REWARD_FUNCTIONS_AVAILABLE = False

class TradingEnv(BaseEnvironment, gym.Env):
    """
    Môi trường giao dịch dựa trên OpenAI Gym.
    Cho phép agent tương tác với dữ liệu thị trường và thực hiện các hành động giao dịch.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array', 'console']}
    
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        data_path: Optional[Union[str, Path]] = None,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        initial_balance: float = 10000.0,
        leverage: float = 1.0,
        max_positions: int = 5,
        fee_rate: float = 0.001,
        window_size: int = 100,
        random_start: bool = True,
        reward_function: str = "profit",
        risk_free_rate: float = 0.0,
        features: Optional[List[str]] = None,
        include_positions: bool = True,
        include_balance: bool = True,
        data_format: str = "ohlcv",
        render_mode: str = "console",
        stoploss_percentage: Optional[float] = None,
        takeprofit_percentage: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
        # Thêm các tham số mới cho target/action
        generate_target_labels: bool = True,
        target_type: str = "price_movement",  # price_movement, binary, multi_class
        target_lookforward: int = 10,  # Số nến nhìn trước
        target_threshold: float = 0.01,  # Ngưỡng % thay đổi giá để xác định nhãn
        target_column: str = "target",  # Tên cột target
        include_timestamp: bool = True,  # Thêm tham số này
        **kwargs
    ):
        """
        Khởi tạo môi trường giao dịch.
        
        Args:
            data: DataFrame chứa dữ liệu thị trường
            data_path: Đường dẫn file dữ liệu nếu data không được cung cấp
            symbol: Cặp giao dịch
            timeframe: Khung thời gian
            initial_balance: Số dư ban đầu
            leverage: Đòn bẩy
            max_positions: Số vị thế tối đa có thể mở
            fee_rate: Tỷ lệ phí giao dịch (0.001 = 0.1%)
            window_size: Kích thước cửa sổ dữ liệu (bao nhiêu nến trong một quan sát)
            random_start: Bắt đầu tại vị trí ngẫu nhiên khi reset
            reward_function: Tên hàm phần thưởng ('profit', 'risk_adjusted', 'custom', etc.)
            risk_free_rate: Lãi suất phi rủi ro (cho reward_function='risk_adjusted')
            features: Danh sách các tính năng để sử dụng từ dữ liệu (None để sử dụng tất cả)
            include_positions: Bao gồm thông tin vị thế trong không gian quan sát
            include_balance: Bao gồm thông tin số dư trong không gian quan sát
            data_format: Định dạng dữ liệu ('ohlcv', 'ohlcv_with_features')
            render_mode: Chế độ hiển thị ('console', 'human', 'rgb_array')
            stoploss_percentage: Phần trăm dừng lỗ tự động (None để không sử dụng)
            takeprofit_percentage: Phần trăm chốt lời tự động (None để không sử dụng)
            logger: Logger tùy chỉnh
        """
        # Gọi constructor của lớp cơ sở
        super().__init__(
            data=data,
            data_path=data_path,
            symbol=symbol,
            timeframe=timeframe,
            initial_balance=initial_balance,
            max_positions=max_positions,
            fee_rate=fee_rate,
            window_size=window_size,
            random_start=random_start,
            reward_function=reward_function,
            logger=logger,
            **kwargs
        )
        
        # Các thuộc tính bổ sung
        self.leverage = leverage
        self.risk_free_rate = risk_free_rate
        self.features = features
        self.include_positions = include_positions
        self.include_balance = include_balance
        self.data_format = data_format
        self.render_mode = render_mode
        self.stoploss_percentage = stoploss_percentage
        self.takeprofit_percentage = takeprofit_percentage
        self.generate_target_labels = generate_target_labels
        self.target_type = target_type
        self.target_lookforward = target_lookforward
        self.target_threshold = target_threshold
        self.target_column = target_column
        self.include_target = kwargs.get('include_target', True)  # Mặc định bật
        if self.target_type == "multi_class":
            self.target_dim = 5  # 5 lớp cho multi_class
        else:
            self.target_dim = 1  # 1 chiều cho price_movement và binary        
        
        # Xác định các cột đặc trưng cần sử dụng
        if self.features is None:
            # Sử dụng tất cả các cột trừ các cột thời gian
            self.feature_columns = [col for col in self.data.columns if col not in ['timestamp', 'date', 'time']]
        else:
            # Sử dụng các cột được chỉ định
            self.feature_columns = self.features
            # Đảm bảo có ít nhất các cột OHLCV
            if data_format == 'ohlcv' or data_format == 'ohlcv_with_features':
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    if col not in self.feature_columns:
                        self.feature_columns.append(col)
        
        # Kiểm tra các cột cần thiết
        for col in self.feature_columns:
            if col not in self.data.columns:
                raise ValueError(f"Cột '{col}' không có trong dữ liệu")
        
        # Thiết lập không gian quan sát
        self.observation_space_manager = ObservationSpace(
            feature_columns=self.feature_columns,
            window_size=self.window_size,
            include_positions=self.include_positions,
            include_balance=self.include_balance,
            max_positions=self.max_positions,
            include_target=hasattr(self, 'include_target') and self.include_target,
            target_dim=getattr(self, 'target_dim', 1)
        )
        
        # Thiết lập không gian hành động
        self.action_space_manager = ActionSpace(
            max_positions=self.max_positions,
            action_type="discrete"  # Hoặc "continuous" tùy thuộc vào nhu cầu
        )
        
        # Thiết lập không gian quan sát và hành động cho gym.Env
        self.observation_space = self.observation_space_manager.get_observation_space()
        self.action_space = self.action_space_manager.get_action_space()
        
        # Khởi tạo trạng thái bổ sung
        self.open_positions = []  # Danh sách các vị thế mở
        self.order_history = []   # Lịch sử các lệnh
        self.position_count = 0   # Số lượng vị thế đã mở
        
        # Theo dõi hiệu suất
        self.performance_metrics = {
            "total_pnl": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "trade_count": 0,
            "max_drawdown": 0.0,
            "max_balance": initial_balance,
            "min_balance": initial_balance
        }
        
        # Biểu đồ cho render
        self.fig = None
        self.ax = None
        
        self.logger.info(f"Đã khởi tạo môi trường giao dịch với {len(self.data)} dòng dữ liệu, {len(self.feature_columns)} đặc trưng")
    
    def reset(self) -> np.ndarray:
        """
        Đặt lại môi trường về trạng thái ban đầu.
        
        Returns:
            Quan sát ban đầu
        """
        # Tạo nhãn target nếu cần
        self._generate_target_labels()

        # Đặt lại vị trí bắt đầu
        if self.random_start:
            self.current_idx = np.random.randint(self.window_size, len(self.data) - 1)
        else:
            self.current_idx = self.window_size
        
        # Đặt lại trạng thái tài chính
        self.current_balance = self.initial_balance
        self.current_nav = self.initial_balance
        self.current_pnl = 0.0
        
        # Đặt lại vị thế
        self.open_positions = []
        self.position_count = 0
        self.current_positions = []
        
        # Đặt lại lịch sử
        self.history = {
            "actions": [],
            "rewards": [],
            "balances": [self.initial_balance],
            "positions": [[]],
            "pnls": [0.0],
            "navs": [self.initial_balance]
        }
        
        # Đặt lại biểu đồ
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        
        # Đặt lại các chỉ số hiệu suất
        self.performance_metrics = {
            "total_pnl": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "trade_count": 0,
            "max_drawdown": 0.0,
            "max_balance": self.initial_balance,
            "min_balance": self.initial_balance
        }
        
        # Trả về quan sát ban đầu
        observation = self.get_observation()
        return observation
    
    def _generate_target_labels(self) -> None:
        """
        Tạo nhãn target dựa trên sự thay đổi giá tương lai.
        Kết quả sẽ được lưu vào một cột mới trong self.data.
        """
        if not hasattr(self, 'generate_target_labels') or not self.generate_target_labels:
            return
        
        if not hasattr(self, 'target_type'):
            self.target_type = "price_movement"
        
        if not hasattr(self, 'target_lookforward'):
            self.target_lookforward = 10
        
        if not hasattr(self, 'target_threshold'):
            self.target_threshold = 0.01
        
        if not hasattr(self, 'target_column'):
            self.target_column = "target"
        
        # Nếu cột target đã tồn tại, không tạo lại
        if self.target_column in self.data.columns:
            self.logger.debug(f"Cột target '{self.target_column}' đã tồn tại, bỏ qua bước tạo target")
            return
        
        self.logger.info(f"Tạo nhãn target với kiểu '{self.target_type}', lookforward={self.target_lookforward}, threshold={self.target_threshold}")
        
        # Lấy dữ liệu giá đóng cửa
        close_prices = self.data['close']
        
        if self.target_type == "price_movement":
            # Tính % thay đổi giá trong tương lai
            future_returns = close_prices.shift(-self.target_lookforward) / close_prices - 1
            
            # Tạo nhãn: 1 (mua) nếu lợi nhuận > threshold, -1 (bán) nếu lỗ > threshold, 0 (giữ nguyên) nếu khác
            self.data[self.target_column] = 0
            self.data.loc[future_returns > self.target_threshold, self.target_column] = 1  # Long signal
            self.data.loc[future_returns < -self.target_threshold, self.target_column] = -1  # Short signal
            
            # Thêm cột future_return để tiện phân tích
            self.data['future_return'] = future_returns
            
        elif self.target_type == "binary":
            # Tính % thay đổi giá trong tương lai
            future_returns = close_prices.shift(-self.target_lookforward) / close_prices - 1
            
            # Tạo nhãn nhị phân: 1 (mua) nếu lợi nhuận > threshold, 0 (không mua) nếu khác
            self.data[self.target_column] = 0
            self.data.loc[future_returns > self.target_threshold, self.target_column] = 1
            
            # Thêm cột future_return để tiện phân tích
            self.data['future_return'] = future_returns
            
        elif self.target_type == "multi_class":
            # Tính % thay đổi giá trong tương lai
            future_returns = close_prices.shift(-self.target_lookforward) / close_prices - 1
            
            # Tạo nhãn đa lớp:
            # 0: Giảm mạnh (< -threshold*2)
            # 1: Giảm nhẹ (-threshold*2 <= x < -threshold)
            # 2: Đi ngang (-threshold <= x <= threshold)
            # 3: Tăng nhẹ (threshold < x <= threshold*2)
            # 4: Tăng mạnh (> threshold*2)
            self.data[self.target_column] = 2  # Mặc định là đi ngang
            self.data.loc[future_returns < -self.target_threshold*2, self.target_column] = 0  # Giảm mạnh
            self.data.loc[(future_returns >= -self.target_threshold*2) & (future_returns < -self.target_threshold), self.target_column] = 1  # Giảm nhẹ
            self.data.loc[(future_returns > self.target_threshold) & (future_returns <= self.target_threshold*2), self.target_column] = 3  # Tăng nhẹ
            self.data.loc[future_returns > self.target_threshold*2, self.target_column] = 4  # Tăng mạnh
            
            # Thêm cột future_return để tiện phân tích
            self.data['future_return'] = future_returns
        
        else:
            self.logger.warning(f"Kiểu target không hợp lệ: {self.target_type}")
            return
        
        # Bỏ qua các hàng cuối không có dữ liệu nhãn
        self.data.loc[self.data.index[-self.target_lookforward:], self.target_column] = np.nan
        
        self.logger.info(f"Đã tạo thành công nhãn target '{self.target_column}' với phân bố: {self.data[self.target_column].value_counts(dropna=True)}") 

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Thực hiện một bước trong môi trường.
        
        Args:
            action: Hành động được đưa ra
            
        Returns:
            Tuple (observation, reward, done, info)
        """
        # Lưu trạng thái trước khi cập nhật
        prev_balance = self.current_balance
        prev_nav = self.current_nav
        
        # Giải mã và thực hiện hành động
        info = self.update_state(action)
        
        # Tính toán phần thưởng
        reward = self.calculate_reward(action)
        
        # Cập nhật lịch sử
        self.history["actions"].append(action)
        self.history["rewards"].append(reward)
        self.history["balances"].append(self.current_balance)
        self.history["positions"].append([pos.copy() for pos in self.open_positions])
        self.history["pnls"].append(self.current_pnl)
        self.history["navs"].append(self.current_nav)
        
        # Kiểm tra số dư âm (phá sản)
        done = self.current_balance <= 0
        
        # Kiểm tra kết thúc dữ liệu
        if self.current_idx >= len(self.data) - 1:
            done = True
        else:
            # Di chuyển đến nến tiếp theo
            self.current_idx += 1
        
        # Cập nhật giá trị tài sản ròng
        self._update_nav()
        
        # Cập nhật các chỉ số hiệu suất
        self._update_performance_metrics(prev_balance)
        
        # Áp dụng stoploss và takeprofit nếu được kích hoạt
        if self.stoploss_percentage is not None or self.takeprofit_percentage is not None:
            self._check_stop_orders()
        
        # Lấy quan sát mới
        observation = self.get_observation()
        
        # Thông tin bổ sung
        info.update({
            "current_idx": self.current_idx,
            "balance": self.current_balance,
            "nav": self.current_nav,
            "reward": reward,
            "pnl": self.current_pnl,
            "positions": len(self.open_positions),
            "timestamp": self.get_current_timestamp(),
            "current_price": self.get_current_price(),
            "performance": self.performance_metrics.copy()
        })
        
        return observation, reward, done, info
    
    def render(self, mode: str = 'human') -> Any:
        """
        Hiển thị trạng thái hiện tại của môi trường.
        
        Args:
            mode: Chế độ hiển thị ('console', 'human', 'rgb_array')
            
        Returns:
            Kết quả hiển thị (tùy thuộc vào mode)
        """
        if mode == 'console' or self.render_mode == 'console':
            # Hiển thị trạng thái dưới dạng văn bản trên console
            current_time = self.get_current_timestamp()
            current_price = self.get_current_price()
            
            output = f"===== Trạng thái tại {current_time} =====\n"
            output += f"Giá: {current_price:.2f}\n"
            output += f"Số dư: {self.current_balance:.2f}\n"
            output += f"NAV: {self.current_nav:.2f}\n"
            output += f"P&L: {self.current_pnl:.2f}\n"
            output += f"Vị thế mở: {len(self.open_positions)}\n"
            
            if self.open_positions:
                output += "-----Danh sách vị thế-----\n"
                for i, pos in enumerate(self.open_positions):
                    side = "Long" if pos['side'] == PositionSide.LONG.value else "Short"
                    profit = (current_price - pos['entry_price']) if pos['side'] == PositionSide.LONG.value else (pos['entry_price'] - current_price)
                    profit *= pos['size']
                    output += f"#{i+1} {side} {pos['size']} @ {pos['entry_price']:.2f} (P&L: {profit:.2f})\n"
            
            print(output)
            return output
            
        elif mode == 'human' or self.render_mode == 'human':
            # Hiển thị biểu đồ tương tác
            if self.fig is None:
                self.fig, self.ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
                plt.ion()  # Kích hoạt chế độ tương tác
            
            # Xóa dữ liệu cũ
            self.ax[0].clear()
            self.ax[1].clear()
            
            # Vẽ biểu đồ giá
            start_idx = max(0, self.current_idx - 100)
            end_idx = min(len(self.data), self.current_idx + 1)
            plot_data = self.data.iloc[start_idx:end_idx]
            
            # Vẽ nến
            self.ax[0].plot(plot_data.index, plot_data['close'], label='Close')
            
            # Đánh dấu các vị thế
            for pos in self.open_positions:
                entry_idx = pos['entry_index']
                if start_idx <= entry_idx <= end_idx:
                    if pos['side'] == PositionSide.LONG.value:
                        self.ax[0].scatter(entry_idx, pos['entry_price'], marker='^', color='green', s=100)
                    else:
                        self.ax[0].scatter(entry_idx, pos['entry_price'], marker='v', color='red', s=100)
            
            # Vẽ biểu đồ số dư
            balances = np.array(self.history["balances"])
            navs = np.array(self.history["navs"])
            self.ax[1].plot(balances, label='Balance')
            self.ax[1].plot(navs, label='NAV')
            
            # Thiết lập nhãn
            self.ax[0].set_title(f"{self.symbol} - {self.timeframe}")
            self.ax[0].set_ylabel("Giá")
            self.ax[0].legend()
            
            self.ax[1].set_title("Số dư & NAV")
            self.ax[1].set_xlabel("Bước")
            self.ax[1].set_ylabel("Giá trị")
            self.ax[1].legend()
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            
            return self.fig
            
        elif mode == 'rgb_array' or self.render_mode == 'rgb_array':
            # Trả về mảng RGB của hình ảnh
            if self.fig is None:
                self.render(mode='human')
            
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img
        
        else:
            raise ValueError(f"Chế độ hiển thị không hợp lệ: {mode}")
    
    def close(self) -> None:
        """
        Đóng môi trường và giải phóng tài nguyên.
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        
        # Ghi log thông tin hiệu suất
        self.logger.info(f"Đóng môi trường với số dư cuối cùng: {self.current_balance:.2f}")
        self.logger.info(f"Tổng P&L: {self.performance_metrics['total_pnl']:.2f}")
        self.logger.info(f"Số giao dịch: {self.performance_metrics['trade_count']}, Thắng: {self.performance_metrics['win_count']}, Thua: {self.performance_metrics['loss_count']}")
        
        # Đóng các tài nguyên khác nếu cần
    
    def get_observation(self) -> np.ndarray:
        """
        Lấy quan sát hiện tại.
        
        Returns:
            Mảng NumPy chứa quan sát
        """
        # Lấy cửa sổ dữ liệu hiện tại
        start_idx = self.current_idx - self.window_size + 1
        end_idx = self.current_idx + 1

        feature_columns = self.feature_columns.copy()
        
        # Nếu cần giữ lại timestamp, thêm vào danh sách cột
        if hasattr(self, 'include_timestamp') and self.include_timestamp and 'timestamp' in self.data.columns:
            if 'timestamp' not in feature_columns:
                feature_columns = ['timestamp'] + feature_columns
                
        if start_idx < 0:
            # Padding nếu không đủ dữ liệu
            padding_size = abs(start_idx)
            start_idx = 0
            window_data = self.data.iloc[start_idx:end_idx][self.feature_columns].values
            padding = np.zeros((padding_size, len(self.feature_columns)))
            window_data = np.vstack([padding, window_data])
        else:
            window_data = self.data.iloc[start_idx:end_idx][self.feature_columns].values
        
        # Lấy thông tin vị thế nếu cần
        position_info = []
        if self.include_positions:
            for i in range(self.max_positions):
                if i < len(self.open_positions):
                    pos = self.open_positions[i]
                    # [side, size, entry_price, entry_time]
                    pos_info = [
                        1.0 if pos['side'] == PositionSide.LONG.value else -1.0,
                        pos['size'] / self.initial_balance,  # Chuẩn hóa kích thước
                        pos['entry_price'] / self.get_current_price(),  # Chuẩn hóa giá
                        (self.current_idx - pos['entry_index']) / self.window_size  # Chuẩn hóa thời gian
                    ]
                else:
                    pos_info = [0.0, 0.0, 0.0, 0.0]  # Không có vị thế
                
                position_info.extend(pos_info)
        
        # Lấy thông tin số dư nếu cần
        balance_info = []
        if self.include_balance:
            # Chuẩn hóa số dư
            balance_info = [self.current_balance / self.initial_balance]
        
        # Thêm target vào quan sát nếu được yêu cầu
        target_info = []
        if hasattr(self, 'include_target') and self.include_target and hasattr(self, 'target_column'):
            if self.target_column in self.data.columns:
                current_target = self.data.iloc[self.current_idx][self.target_column]
                if np.isnan(current_target):
                    target_info = [0.0]  # Nếu không có giá trị, mặc định là 0
                else:
                    # Chuẩn hóa target nếu cần
                    if self.target_type == "price_movement" or self.target_type == "binary":
                        target_info = [float(current_target)]
                    elif self.target_type == "multi_class":
                        # One-hot encoding cho target đa lớp
                        num_classes = 5  # 0, 1, 2, 3, 4
                        one_hot = np.zeros(num_classes)
                        one_hot[int(current_target)] = 1.0
                        target_info = one_hot.tolist()
        
        # Kết hợp tất cả thông tin
        observation = self.observation_space_manager.create_observation(
            window_data=window_data,
            position_data=np.array(position_info) if position_info else None,
            balance_data=np.array(balance_info) if balance_info else None,
            target_data=np.array(target_info) if target_info else None  # Thêm dòng này
        )
        
        return observation
    
    def update_state(self, action: Any) -> Dict[str, Any]:
        """
        Cập nhật trạng thái dựa trên hành động.
        
        Args:
            action: Hành động được đưa ra
            
        Returns:
            Thông tin về cập nhật
        """
        # Giải mã hành động
        action_info = self.action_space_manager.decode_action(action)
        action_type = action_info.get('action_type')
        
        # Lấy giá hiện tại
        current_price = self.get_current_price()
        
        info = {
            "action": action_info,
            "price": current_price,
            "timestamp": self.get_current_timestamp(),
            "success": False,
            "reason": "",
            "pnl": 0.0
        }
        
        # Xử lý các loại hành động
        if action_type == 'buy':
            # Mở vị thế Long
            position_size = action_info.get('size', 0.1) * self.current_balance
            
            # Kiểm tra xem có đủ số dư không
            if position_size <= 0 or position_size > self.current_balance:
                info["reason"] = "Kích thước vị thế không hợp lệ"
                return info
            
            # Kiểm tra xem có đạt giới hạn vị thế không
            if len(self.open_positions) >= self.max_positions:
                info["reason"] = "Đã đạt giới hạn vị thế tối đa"
                return info
            
            # Tính phí giao dịch
            fee = position_size * self.fee_rate
            
            # Cập nhật số dư
            self.current_balance -= fee
            
            # Mở vị thế mới
            new_position = {
                'id': self.position_count,
                'side': PositionSide.LONG.value,
                'size': position_size,
                'entry_price': current_price,
                'entry_index': self.current_idx,
                'entry_time': self.get_current_timestamp(),
                'leverage': self.leverage,
                'fee': fee,
                'stoploss': None if self.stoploss_percentage is None else current_price * (1 - self.stoploss_percentage),
                'takeprofit': None if self.takeprofit_percentage is None else current_price * (1 + self.takeprofit_percentage)
            }
            
            self.open_positions.append(new_position)
            self.position_count += 1
            
            info["success"] = True
            info["position_id"] = new_position['id']
            
        elif action_type == 'sell':
            # Mở vị thế Short
            position_size = action_info.get('size', 0.1) * self.current_balance
            
            # Kiểm tra xem có đủ số dư không
            if position_size <= 0 or position_size > self.current_balance:
                info["reason"] = "Kích thước vị thế không hợp lệ"
                return info
            
            # Kiểm tra xem có đạt giới hạn vị thế không
            if len(self.open_positions) >= self.max_positions:
                info["reason"] = "Đã đạt giới hạn vị thế tối đa"
                return info
            
            # Tính phí giao dịch
            fee = position_size * self.fee_rate
            
            # Cập nhật số dư
            self.current_balance -= fee
            
            # Mở vị thế mới
            new_position = {
                'id': self.position_count,
                'side': PositionSide.SHORT.value,
                'size': position_size,
                'entry_price': current_price,
                'entry_index': self.current_idx,
                'entry_time': self.get_current_timestamp(),
                'leverage': self.leverage,
                'fee': fee,
                'stoploss': None if self.stoploss_percentage is None else current_price * (1 + self.stoploss_percentage),
                'takeprofit': None if self.takeprofit_percentage is None else current_price * (1 - self.takeprofit_percentage)
            }
            
            self.open_positions.append(new_position)
            self.position_count += 1
            
            info["success"] = True
            info["position_id"] = new_position['id']
            
        elif action_type == 'close':
            # Đóng vị thế cụ thể
            position_id = action_info.get('position_id')
            
            # Tìm vị thế cần đóng
            position_idx = None
            for i, pos in enumerate(self.open_positions):
                if pos['id'] == position_id:
                    position_idx = i
                    break
            
            if position_idx is None:
                info["reason"] = f"Không tìm thấy vị thế ID {position_id}"
                return info
            
            # Đóng vị thế
            position = self.open_positions.pop(position_idx)
            pnl = self._calculate_position_pnl(position, current_price)
            
            # Tính phí giao dịch đóng vị thế
            close_fee = position['size'] * self.fee_rate
            
            # Cập nhật số dư
            self.current_balance += position['size'] + pnl - close_fee
            self.current_pnl += pnl
            
            # Cập nhật lịch sử giao dịch
            self.order_history.append({
                'position_id': position['id'],
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'size': position['size'],
                'entry_time': position['entry_time'],
                'exit_time': self.get_current_timestamp(),
                'pnl': pnl,
                'fee': position['fee'] + close_fee
            })
            
            # Cập nhật hiệu suất
            self.performance_metrics["trade_count"] += 1
            if pnl > 0:
                self.performance_metrics["win_count"] += 1
            else:
                self.performance_metrics["loss_count"] += 1
            
            self.performance_metrics["total_pnl"] += pnl
            
            info["success"] = True
            info["pnl"] = pnl
            
        elif action_type == 'close_all':
            # Đóng tất cả vị thế
            if not self.open_positions:
                info["reason"] = "Không có vị thế nào để đóng"
                return info
            
            total_pnl = 0.0
            total_fee = 0.0
            
            # Đóng từng vị thế
            for pos in self.open_positions[:]:
                pnl = self._calculate_position_pnl(pos, current_price)
                close_fee = pos['size'] * self.fee_rate
                
                total_pnl += pnl
                total_fee += close_fee
                
                # Cập nhật lịch sử giao dịch
                self.order_history.append({
                    'position_id': pos['id'],
                    'side': pos['side'],
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'size': pos['size'],
                    'entry_time': pos['entry_time'],
                    'exit_time': self.get_current_timestamp(),
                    'pnl': pnl,
                    'fee': pos['fee'] + close_fee
                })
                
                # Cập nhật hiệu suất
                self.performance_metrics["trade_count"] += 1
                if pnl > 0:
                    self.performance_metrics["win_count"] += 1
                else:
                    self.performance_metrics["loss_count"] += 1
            
            # Cập nhật số dư
            total_size = sum(pos['size'] for pos in self.open_positions)
            self.current_balance += total_size + total_pnl - total_fee
            self.current_pnl += total_pnl
            
            # Xóa tất cả vị thế
            self.open_positions = []
            
            self.performance_metrics["total_pnl"] += total_pnl
            
            info["success"] = True
            info["pnl"] = total_pnl
            
        elif action_type == 'hold':
            # Không làm gì
            info["success"] = True
            
        else:
            info["reason"] = f"Loại hành động không hợp lệ: {action_type}"
        
        # Cập nhật NAV
        self._update_nav()
        
        return info
    
    def calculate_reward(self, action: Any) -> float:
        """
        Tính toán phần thưởng cho hành động.
        
        Args:
            action: Hành động được đưa ra
            
        Returns:
            Giá trị phần thưởng
        """
        # Kiểm tra xem các module reward_functions có sẵn không
        if not REWARD_FUNCTIONS_AVAILABLE:
            # Tính toán phần thưởng đơn giản dựa trên thay đổi NAV
            if len(self.history["navs"]) < 2:
                return 0.0
            
            prev_nav = self.history["navs"][-2]
            curr_nav = self.history["navs"][-1]
            
            # Phần thưởng là % thay đổi NAV
            return (curr_nav - prev_nav) / prev_nav
        
        # Sử dụng các hàm phần thưởng từ module
        if self.reward_function_name == RewardFunction.PROFIT.value:
            return calculate_profit_reward(
                self.history["navs"],
                self.history["balances"],
                self.history["positions"],
                self.current_pnl
            )
        elif self.reward_function_name == RewardFunction.RISK_ADJUSTED.value:
            return calculate_risk_adjusted_reward(
                self.history["navs"],
                self.history["balances"],
                self.history["positions"],
                self.current_pnl,
                self.risk_free_rate
            )
        elif self.reward_function_name == RewardFunction.CUSTOM.value:
            return calculate_custom_reward(
                self.history["navs"],
                self.history["balances"],
                self.history["positions"],
                self.current_pnl,
                self.performance_metrics
            )
        else:
            # Mặc định, sử dụng phần thưởng dựa trên thay đổi NAV
            if len(self.history["navs"]) < 2:
                return 0.0
            
            prev_nav = self.history["navs"][-2]
            curr_nav = self.history["navs"][-1]
            
            # Phần thưởng là % thay đổi NAV
            return (curr_nav - prev_nav) / prev_nav
    
    def _update_nav(self) -> None:
        """
        Cập nhật giá trị tài sản ròng.
        """
        # NAV = Số dư + Giá trị các vị thế mở
        nav = self.current_balance
        
        if self.open_positions:
            current_price = self.get_current_price()
            for pos in self.open_positions:
                pnl = self._calculate_position_pnl(pos, current_price)
                nav += pos['size'] + pnl
        
        self.current_nav = nav
    
    def _calculate_position_pnl(self, position: Dict[str, Any], current_price: float) -> float:
        """
        Tính lãi/lỗ của một vị thế.
        
        Args:
            position: Thông tin vị thế
            current_price: Giá hiện tại
            
        Returns:
            Giá trị lãi/lỗ
        """
        size = position['size']
        entry_price = position['entry_price']
        leverage = position['leverage']
        
        if position['side'] == PositionSide.LONG.value:
            # Đối với vị thế Long: P&L = size * (current_price - entry_price) / entry_price * leverage
            return size * (current_price - entry_price) / entry_price * leverage
        else:
            # Đối với vị thế Short: P&L = size * (entry_price - current_price) / entry_price * leverage
            return size * (entry_price - current_price) / entry_price * leverage
    
    def _update_performance_metrics(self, prev_balance: float) -> None:
        """
        Cập nhật các chỉ số hiệu suất.
        
        Args:
            prev_balance: Số dư trước đó
        """
        # Cập nhật balance tối đa/tối thiểu
        if self.current_balance > self.performance_metrics["max_balance"]:
            self.performance_metrics["max_balance"] = self.current_balance
        
        if self.current_balance < self.performance_metrics["min_balance"]:
            self.performance_metrics["min_balance"] = self.current_balance
        
        # Tính drawdown
        max_balance = self.performance_metrics["max_balance"]
        current_drawdown = (max_balance - self.current_balance) / max_balance if max_balance > 0 else 0
        
        if current_drawdown > self.performance_metrics["max_drawdown"]:
            self.performance_metrics["max_drawdown"] = current_drawdown
    
    def _check_stop_orders(self) -> None:
        """
        Kiểm tra và thực hiện các lệnh stop loss và take profit.
        """
        if not self.open_positions:
            return
        
        current_price = self.get_current_price()
        positions_to_close = []
        
        # Kiểm tra từng vị thế
        for i, pos in enumerate(self.open_positions):
            if pos['side'] == PositionSide.LONG.value:
                # Đối với vị thế Long
                if pos['stoploss'] is not None and current_price <= pos['stoploss']:
                    # Kích hoạt stoploss
                    positions_to_close.append((i, "stoploss"))
                elif pos['takeprofit'] is not None and current_price >= pos['takeprofit']:
                    # Kích hoạt takeprofit
                    positions_to_close.append((i, "takeprofit"))
            else:
                # Đối với vị thế Short
                if pos['stoploss'] is not None and current_price >= pos['stoploss']:
                    # Kích hoạt stoploss
                    positions_to_close.append((i, "stoploss"))
                elif pos['takeprofit'] is not None and current_price <= pos['takeprofit']:
                    # Kích hoạt takeprofit
                    positions_to_close.append((i, "takeprofit"))
        
        # Đóng các vị thế theo thứ tự ngược
        for i, reason in sorted(positions_to_close, reverse=True):
            position = self.open_positions.pop(i)
            pnl = self._calculate_position_pnl(position, current_price)
            
            # Tính phí giao dịch đóng vị thế
            close_fee = position['size'] * self.fee_rate
            
            # Cập nhật số dư
            self.current_balance += position['size'] + pnl - close_fee
            self.current_pnl += pnl
            
            # Cập nhật lịch sử giao dịch
            self.order_history.append({
                'position_id': position['id'],
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'size': position['size'],
                'entry_time': position['entry_time'],
                'exit_time': self.get_current_timestamp(),
                'pnl': pnl,
                'fee': position['fee'] + close_fee,
                'close_reason': reason
            })
            
            # Cập nhật hiệu suất
            self.performance_metrics["trade_count"] += 1
            if pnl > 0:
                self.performance_metrics["win_count"] += 1
            else:
                self.performance_metrics["loss_count"] += 1
            
            self.performance_metrics["total_pnl"] += pnl
            
            self.logger.info(f"Đóng vị thế theo {reason}: ID {position['id']}, P&L {pnl:.2f}")