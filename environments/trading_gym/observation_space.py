"""
Không gian quan sát cho môi trường giao dịch.
File này định nghĩa lớp ObservationSpace để quản lý việc tạo và chuẩn hóa
không gian quan sát cho agent.
"""

import numpy as np
from gym import spaces
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

from config.logging_config import get_logger

class ObservationSpace:
    """
    Lớp quản lý không gian quan sát trong môi trường giao dịch.
    """
    
    def __init__(
        self,
        feature_columns: List[str],
        window_size: int = 100,
        include_positions: bool = True,
        include_balance: bool = True,
        max_positions: int = 5,
        normalize: bool = True,
        flattened: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo không gian quan sát.
        
        Args:
            feature_columns: Danh sách các cột đặc trưng sử dụng
            window_size: Kích thước cửa sổ dữ liệu
            include_positions: Bao gồm thông tin vị thế trong quan sát
            include_balance: Bao gồm thông tin số dư trong quan sát
            max_positions: Số vị thế tối đa được theo dõi
            normalize: Chuẩn hóa dữ liệu hay không
            flattened: Làm phẳng không gian quan sát hay không
            logger: Logger tùy chỉnh
        """
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.include_positions = include_positions
        self.include_balance = include_balance
        self.max_positions = max_positions
        self.normalize = normalize
        self.flattened = flattened
        self.logger = logger or get_logger("observation_space")
        
        # Thông tin về kích thước
        self.num_features = len(feature_columns)
        self.position_dim = 4 * max_positions if include_positions else 0  # [side, size, entry_price, entry_time]
        self.balance_dim = 1 if include_balance else 0
        
        # Tính toán không gian quan sát
        self._create_observation_space()
        
        self.logger.info(f"Đã khởi tạo ObservationSpace với {self.num_features} đặc trưng, window_size={window_size}")
    
    def _create_observation_space(self) -> None:
        """
        Tạo không gian quan sát dựa trên các tham số.
        """
        if self.flattened:
            # Không gian quan sát dạng phẳng
            total_dim = self.window_size * self.num_features + self.position_dim + self.balance_dim
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
            )
        else:
            # Không gian quan sát dạng ma trận
            spaces_dict = {
                "market": spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(self.window_size, self.num_features), 
                    dtype=np.float32
                )
            }
            
            if self.include_positions:
                spaces_dict["positions"] = spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(self.position_dim,), 
                    dtype=np.float32
                )
            
            if self.include_balance:
                spaces_dict["balance"] = spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(self.balance_dim,), 
                    dtype=np.float32
                )
            
            self.observation_space = spaces.Dict(spaces_dict)
    
    def get_observation_space(self) -> spaces.Space:
        """
        Lấy không gian quan sát đã định nghĩa.
        
        Returns:
            Không gian quan sát gym.spaces
        """
        return self.observation_space
    
    def create_observation(
        self,
        window_data: np.ndarray,
        position_data: Optional[np.ndarray] = None,
        balance_data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Tạo quan sát từ dữ liệu.
        
        Args:
            window_data: Mảng NumPy chứa dữ liệu cửa sổ
            position_data: Mảng NumPy chứa dữ liệu vị thế
            balance_data: Mảng NumPy chứa dữ liệu số dư
            
        Returns:
            Mảng NumPy chứa quan sát
        """
        # Chuẩn hóa dữ liệu nếu cần
        if self.normalize:
            window_data = self._normalize_window(window_data)
        
        if self.flattened:
            # Làm phẳng dữ liệu
            observation = window_data.flatten()
            
            # Thêm thông tin vị thế nếu có
            if self.include_positions and position_data is not None:
                observation = np.concatenate([observation, position_data])
            
            # Thêm thông tin số dư nếu có
            if self.include_balance and balance_data is not None:
                observation = np.concatenate([observation, balance_data])
            
            return observation.astype(np.float32)
        else:
            # Trả về dưới dạng Dict
            observation = {"market": window_data.astype(np.float32)}
            
            if self.include_positions and position_data is not None:
                observation["positions"] = position_data.astype(np.float32)
            
            if self.include_balance and balance_data is not None:
                observation["balance"] = balance_data.astype(np.float32)
            
            return observation
    
    def _normalize_window(self, window_data: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa dữ liệu cửa sổ.
        
        Args:
            window_data: Mảng NumPy chứa dữ liệu cửa sổ
            
        Returns:
            Mảng NumPy đã chuẩn hóa
        """
        # Áp dụng chuẩn hóa z-score cho mỗi cột
        # Chú ý: Ta cần chuẩn hóa theo theo từng cột, không phải toàn bộ mảng
        if len(window_data) <= 1:
            return window_data
        
        normalized_data = np.zeros_like(window_data)
        
        for i in range(window_data.shape[1]):
            series = window_data[:, i]
            mean = np.mean(series)
            std = np.std(series)
            
            if std > 0:
                normalized_data[:, i] = (series - mean) / std
            else:
                normalized_data[:, i] = series - mean
        
        return normalized_data
    
    def denormalize_observation(
        self, 
        observation: np.ndarray, 
        original_window_data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Chuyển đổi quan sát chuẩn hóa về dạng gốc.
        
        Args:
            observation: Quan sát đã chuẩn hóa
            original_window_data: Dữ liệu cửa sổ gốc để lấy thông số chuẩn hóa
            
        Returns:
            Dict chứa quan sát đã giải mã
        """
        if not self.normalize:
            # Nếu không chuẩn hóa, trả về trực tiếp
            if self.flattened:
                # Phân tách các phần
                market_size = self.window_size * self.num_features
                market_data = observation[:market_size].reshape(self.window_size, self.num_features)
                
                result = {"market": market_data}
                
                if self.include_positions:
                    position_start = market_size
                    position_end = position_start + self.position_dim
                    result["positions"] = observation[position_start:position_end]
                
                if self.include_balance:
                    balance_start = market_size + self.position_dim
                    result["balance"] = observation[balance_start:]
                
                return result
            else:
                # Đã ở dạng Dict
                return observation
        
        # Trường hợp đã chuẩn hóa, cần khôi phục
        if self.flattened:
            # Phân tách các phần
            market_size = self.window_size * self.num_features
            
            # Lấy dữ liệu thị trường từ quan sát
            normalized_market = observation[:market_size].reshape(self.window_size, self.num_features)
            
            # Khôi phục dữ liệu thị trường
            denormalized_market = np.zeros_like(normalized_market)
            
            for i in range(normalized_market.shape[1]):
                series = original_window_data[:, i]
                mean = np.mean(series)
                std = np.std(series)
                
                if std > 0:
                    denormalized_market[:, i] = normalized_market[:, i] * std + mean
                else:
                    denormalized_market[:, i] = normalized_market[:, i] + mean
            
            result = {"market": denormalized_market}
            
            # Các phần còn lại không cần khôi phục vì chúng đã được chuẩn hóa theo cách riêng
            if self.include_positions:
                position_start = market_size
                position_end = position_start + self.position_dim
                result["positions"] = observation[position_start:position_end]
            
            if self.include_balance:
                balance_start = market_size + self.position_dim
                result["balance"] = observation[balance_start:]
            
            return result
        else:
            # Trường hợp Dict
            normalized_market = observation["market"]
            
            # Khôi phục dữ liệu thị trường
            denormalized_market = np.zeros_like(normalized_market)
            
            for i in range(normalized_market.shape[1]):
                series = original_window_data[:, i]
                mean = np.mean(series)
                std = np.std(series)
                
                if std > 0:
                    denormalized_market[:, i] = normalized_market[:, i] * std + mean
                else:
                    denormalized_market[:, i] = normalized_market[:, i] + mean
            
            result = {"market": denormalized_market}
            
            # Giữ nguyên các phần còn lại
            if "positions" in observation:
                result["positions"] = observation["positions"]
            
            if "balance" in observation:
                result["balance"] = observation["balance"]
            
            return result