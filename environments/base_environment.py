"""
Lớp môi trường cơ sở.
File này định nghĩa lớp BaseEnvironment làm nền tảng cho các môi trường giao dịch,
cung cấp các phương thức và thuộc tính cơ bản mà tất cả các môi trường cần có.
"""

import os
import gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod
import logging

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import PositionSide, OrderType, OrderStatus

class BaseEnvironment(ABC):
    """
    Lớp cơ sở cho tất cả các môi trường giao dịch.
    Định nghĩa giao diện chung mà tất cả các môi trường phải tuân theo.
    """
    
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        data_path: Optional[Union[str, Path]] = None,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        initial_balance: float = 10000.0,
        max_positions: int = 5,
        fee_rate: float = 0.001,
        window_size: int = 100,
        random_start: bool = True,
        reward_function: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Khởi tạo môi trường cơ sở.
        
        Args:
            data: DataFrame chứa dữ liệu thị trường
            data_path: Đường dẫn file dữ liệu nếu data không được cung cấp
            symbol: Cặp giao dịch
            timeframe: Khung thời gian
            initial_balance: Số dư ban đầu
            max_positions: Số vị thế tối đa có thể mở
            fee_rate: Tỷ lệ phí giao dịch (0.001 = 0.1%)
            window_size: Kích thước cửa sổ dữ liệu (bao nhiêu nến trong một quan sát)
            random_start: Bắt đầu tại vị trí ngẫu nhiên khi reset
            reward_function: Tên hàm phần thưởng ('profit', 'risk_adjusted', 'custom', etc.)
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("environment")
        
        # Tải dữ liệu
        self.data = self._load_data(data, data_path)
        
        # Kiểm tra dữ liệu
        if self.data is None or len(self.data) < window_size:
            raise ValueError(f"Dữ liệu không đủ, cần ít nhất {window_size} dòng")
        
        # Thiết lập các thuộc tính
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.max_positions = max_positions
        self.fee_rate = fee_rate
        self.window_size = window_size
        self.random_start = random_start
        self.reward_function_name = reward_function or "profit"
        
        # Chỉ số hiện tại trong dữ liệu
        self.current_idx = 0
        
        # Trạng thái hiện tại (sẽ được cập nhật trong reset)
        self.current_balance = initial_balance
        self.current_positions = []
        self.current_pnl = 0.0
        self.current_nav = initial_balance
        
        # Lịch sử các hành động và trạng thái
        self.history = {
            "actions": [],
            "rewards": [],
            "balances": [],
            "positions": [],
            "pnls": [],
            "navs": []
        }
        
        # Không gian hành động và quan sát (sẽ được định nghĩa bởi các lớp con)
        self.action_space = None
        self.observation_space = None
        
        # Siêu tham số bổ sung
        self.kwargs = kwargs
        
        self.logger.info(f"Đã khởi tạo môi trường cơ sở với {len(self.data)} dòng dữ liệu")
    
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
            self.logger.warning("Không có dữ liệu được cung cấp")
            return None
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Đặt lại môi trường về trạng thái ban đầu.
        
        Returns:
            Quan sát ban đầu
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Thực hiện một bước trong môi trường.
        
        Args:
            action: Hành động được đưa ra
            
        Returns:
            Tuple (observation, reward, done, info)
        """
        pass
    
    @abstractmethod
    def render(self, mode: str = 'human') -> Any:
        """
        Hiển thị trạng thái hiện tại của môi trường.
        
        Args:
            mode: Chế độ hiển thị
            
        Returns:
            Kết quả hiển thị (tùy thuộc vào mode)
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Đóng môi trường và giải phóng tài nguyên.
        """
        pass
    
    def get_observation(self) -> np.ndarray:
        """
        Lấy quan sát hiện tại.
        
        Returns:
            Mảng NumPy chứa quan sát
        """
        # Phương thức này sẽ được ghi đè bởi các lớp con
        raise NotImplementedError("Phương thức get_observation() cần được định nghĩa trong lớp con")
    
    def calculate_reward(self, action: Any) -> float:
        """
        Tính toán phần thưởng cho hành động hiện tại.
        
        Args:
            action: Hành động được đưa ra
            
        Returns:
            Giá trị phần thưởng
        """
        # Phương thức này sẽ được ghi đè bởi các lớp con
        raise NotImplementedError("Phương thức calculate_reward() cần được định nghĩa trong lớp con")
    
    def update_state(self, action: Any) -> Dict[str, Any]:
        """
        Cập nhật trạng thái dựa trên hành động.
        
        Args:
            action: Hành động được đưa ra
            
        Returns:
            Thông tin về trạng thái mới
        """
        # Phương thức này sẽ được ghi đè bởi các lớp con
        raise NotImplementedError("Phương thức update_state() cần được định nghĩa trong lớp con")
    
    def get_current_price(self) -> float:
        """
        Lấy giá hiện tại.
        
        Returns:
            Giá hiện tại
        """
        if self.current_idx < len(self.data):
            return self.data.iloc[self.current_idx]['close']
        else:
            self.logger.warning("Chỉ số hiện tại vượt quá độ dài dữ liệu")
            return self.data.iloc[-1]['close']
    
    def get_current_timestamp(self) -> pd.Timestamp:
        """
        Lấy timestamp hiện tại.
        
        Returns:
            Timestamp hiện tại
        """
        if 'timestamp' in self.data.columns:
            return self.data.iloc[self.current_idx]['timestamp']
        else:
            # Tạo timestamp dựa trên chỉ số nếu không có cột timestamp
            return pd.Timestamp.now() + pd.Timedelta(hours=self.current_idx)
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin trạng thái hiện tại.
        
        Returns:
            Dict chứa thông tin trạng thái
        """
        return {
            "balance": self.current_balance,
            "positions": self.current_positions.copy(),
            "pnl": self.current_pnl,
            "nav": self.current_nav,
            "timestamp": self.get_current_timestamp(),
            "price": self.get_current_price(),
            "index": self.current_idx
        }
    
    def get_history(self) -> Dict[str, List]:
        """
        Lấy lịch sử giao dịch.
        
        Returns:
            Dict chứa lịch sử
        """
        return self.history.copy()
    
    def save_history(self, filepath: Union[str, Path]) -> None:
        """
        Lưu lịch sử giao dịch vào file.
        
        Args:
            filepath: Đường dẫn file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Chuyển đổi lịch sử thành DataFrame
        history_df = pd.DataFrame({
            "balance": self.history["balances"],
            "pnl": self.history["pnls"],
            "nav": self.history["navs"],
            "reward": self.history["rewards"],
        })
        
        # Lưu vào file
        history_df.to_csv(filepath, index=False)
        self.logger.info(f"Đã lưu lịch sử giao dịch vào {filepath}")