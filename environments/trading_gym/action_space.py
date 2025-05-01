"""
Không gian hành động cho môi trường giao dịch.
File này định nghĩa lớp ActionSpace để quản lý việc tạo và giải mã
không gian hành động cho agent.
"""

import numpy as np
from gym import spaces
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

from config.logging_config import get_logger
from config.constants import PositionSide, OrderType

class ActionSpace:
    """
    Lớp quản lý không gian hành động trong môi trường giao dịch.
    """
    
    def __init__(
        self,
        max_positions: int = 5,
        action_type: str = "discrete",
        continuous_actions: bool = False,
        position_sizing: bool = True,
        allow_short: bool = True,
        include_position_actions: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo không gian hành động.
        
        Args:
            max_positions: Số vị thế tối đa
            action_type: Loại hành động ('discrete', 'multi_discrete', 'continuous')
            continuous_actions: Sử dụng hành động liên tục hay không (cho kích thước vị thế)
            position_sizing: Cho phép agent quyết định kích thước vị thế hay không
            allow_short: Cho phép bán khống hay không
            include_position_actions: Bao gồm hành động đóng vị thế cụ thể hay không
            logger: Logger tùy chỉnh
        """
        self.max_positions = max_positions
        self.action_type = action_type
        self.continuous_actions = continuous_actions
        self.position_sizing = position_sizing
        self.allow_short = allow_short
        self.include_position_actions = include_position_actions
        self.logger = logger or get_logger("action_space")
        
        # Tạo không gian hành động
        self._create_action_space()
        
        self.logger.info(f"Đã khởi tạo ActionSpace với loại {action_type}, max_positions={max_positions}")
    
    def _create_action_space(self) -> None:
        """
        Tạo không gian hành động dựa trên các tham số.
        """
        if self.action_type == "discrete":
            # Không gian hành động rời rạc
            num_actions = 1  # Hành động hold
            
            # Thêm hành động mua/bán
            if self.position_sizing:
                # Mỗi hành động mua/bán có nhiều mức kích thước
                num_size_levels = 5  # Ví dụ: 10%, 25%, 50%, 75%, 100% của số dư
                num_actions += 2 * num_size_levels if self.allow_short else num_size_levels
            else:
                # Chỉ có 1 mức kích thước
                num_actions += 2 if self.allow_short else 1
            
            # Thêm hành động đóng tất cả
            num_actions += 1
            
            # Thêm hành động đóng vị thế cụ thể
            if self.include_position_actions:
                num_actions += self.max_positions
            
            self.action_space = spaces.Discrete(num_actions)
            
            # Ánh xạ giữa mã hành động và hành động thực tế
            self.action_map = self._create_discrete_action_map()
            
        elif self.action_type == "multi_discrete":
            # Không gian hành động đa rời rạc
            action_dims = []
            
            # Hành động loại: mua, bán, đóng, giữ
            num_action_types = 4 if self.allow_short else 3
            action_dims.append(num_action_types)
            
            # Kích thước vị thế
            if self.position_sizing:
                action_dims.append(5)  # 5 mức kích thước
            
            # Nếu có hành động đóng vị thế cụ thể
            if self.include_position_actions:
                action_dims.append(self.max_positions + 1)  # +1 cho "không đóng vị thế nào"
            
            self.action_space = spaces.MultiDiscrete(action_dims)
            
        elif self.action_type == "continuous":
            # Không gian hành động liên tục
            if self.include_position_actions:
                # [action_type, size, position_idx]
                action_dims = 3
            else:
                # [action_type, size]
                action_dims = 2
            
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(action_dims,), dtype=np.float32
            )
        
        else:
            raise ValueError(f"Loại hành động không hợp lệ: {self.action_type}")
    
    def _create_discrete_action_map(self) -> Dict[int, Dict[str, Any]]:
        """
        Tạo ánh xạ cho không gian hành động rời rạc.
        
        Returns:
            Dict ánh xạ từ mã hành động sang hành động thực tế
        """
        action_map = {}
        action_idx = 0
        
        # Hành động hold (không làm gì)
        action_map[action_idx] = {"action_type": "hold"}
        action_idx += 1
        
        # Hành động mua
        if self.position_sizing:
            size_levels = [0.1, 0.25, 0.5, 0.75, 1.0]  # % của số dư
            for size in size_levels:
                action_map[action_idx] = {"action_type": "buy", "size": size}
                action_idx += 1
        else:
            action_map[action_idx] = {"action_type": "buy", "size": 1.0}
            action_idx += 1
        
        # Hành động bán (short)
        if self.allow_short:
            if self.position_sizing:
                for size in size_levels:
                    action_map[action_idx] = {"action_type": "sell", "size": size}
                    action_idx += 1
            else:
                action_map[action_idx] = {"action_type": "sell", "size": 1.0}
                action_idx += 1
        
        # Hành động đóng tất cả
        action_map[action_idx] = {"action_type": "close_all"}
        action_idx += 1
        
        # Hành động đóng vị thế cụ thể
        if self.include_position_actions:
            for pos_idx in range(self.max_positions):
                action_map[action_idx] = {"action_type": "close", "position_id": pos_idx}
                action_idx += 1
        
        return action_map
    
    def get_action_space(self) -> spaces.Space:
        """
        Lấy không gian hành động đã định nghĩa.
        
        Returns:
            Không gian hành động gym.spaces
        """
        return self.action_space
    
    def decode_action(self, action: Any) -> Dict[str, Any]:
        """
        Giải mã hành động từ mã hành động.
        
        Args:
            action: Mã hành động từ agent
            
        Returns:
            Dict chứa thông tin hành động đã giải mã
        """
        if self.action_type == "discrete":
            # Kiểm tra hành động có hợp lệ không
            if action < 0 or action >= len(self.action_map):
                self.logger.warning(f"Hành động không hợp lệ: {action}, sử dụng hành động mặc định (hold)")
                return {"action_type": "hold"}
            
            # Trả về hành động từ ánh xạ
            return self.action_map[action]
        
        elif self.action_type == "multi_discrete":
            # Giải mã hành động đa rời rạc
            action_type_idx = action[0]
            
            if action_type_idx == 0:
                # Hold
                return {"action_type": "hold"}
            
            elif action_type_idx == 1:
                # Buy
                if self.position_sizing:
                    size_idx = action[1]
                    size_levels = [0.1, 0.25, 0.5, 0.75, 1.0]
                    size = size_levels[min(size_idx, len(size_levels) - 1)]
                    return {"action_type": "buy", "size": size}
                else:
                    return {"action_type": "buy", "size": 1.0}
            
            elif action_type_idx == 2 and self.allow_short:
                # Sell (short)
                if self.position_sizing:
                    size_idx = action[1]
                    size_levels = [0.1, 0.25, 0.5, 0.75, 1.0]
                    size = size_levels[min(size_idx, len(size_levels) - 1)]
                    return {"action_type": "sell", "size": size}
                else:
                    return {"action_type": "sell", "size": 1.0}
            
            elif (action_type_idx == 2 and not self.allow_short) or (action_type_idx == 3):
                # Close
                if self.include_position_actions:
                    pos_idx = action[-1]
                    if pos_idx == 0:
                        return {"action_type": "close_all"}
                    else:
                        return {"action_type": "close", "position_id": pos_idx - 1}
                else:
                    return {"action_type": "close_all"}
            
            # Mặc định
            return {"action_type": "hold"}
        
        elif self.action_type == "continuous":
            # Giải mã hành động liên tục
            action_type_value = action[0]
            
            # Ánh xạ action_type từ [-1, 1] sang loại hành động
            if action_type_value < -0.5:
                # Sell (short)
                if not self.allow_short:
                    return {"action_type": "hold"}
                
                action_type = "sell"
            elif action_type_value < 0:
                # Close
                if self.include_position_actions and len(action) > 2:
                    # Ánh xạ position_idx từ [-1, 1] sang index
                    pos_value = action[2]
                    pos_idx = int((pos_value + 1) / 2 * self.max_positions)
                    pos_idx = max(0, min(pos_idx, self.max_positions - 1))
                    
                    return {"action_type": "close", "position_id": pos_idx}
                else:
                    return {"action_type": "close_all"}
            elif action_type_value < 0.5:
                # Hold
                return {"action_type": "hold"}
            else:
                # Buy
                action_type = "buy"
            
            # Nếu là buy hoặc sell, lấy kích thước
            if action_type in ["buy", "sell"] and self.position_sizing:
                size_value = action[1]
                # Ánh xạ size từ [-1, 1] sang [0.1, 1.0]
                size = 0.1 + (size_value + 1) / 2 * 0.9
                size = max(0.1, min(size, 1.0))
                
                return {"action_type": action_type, "size": size}
            elif action_type in ["buy", "sell"]:
                return {"action_type": action_type, "size": 1.0}
            
            # Mặc định
            return {"action_type": "hold"}
        
        else:
            self.logger.error(f"Loại hành động không được hỗ trợ: {self.action_type}")
            return {"action_type": "hold"}
    
    def encode_action(self, action_info: Dict[str, Any]) -> Any:
        """
        Mã hóa thông tin hành động thành mã hành động.
        
        Args:
            action_info: Thông tin hành động
            
        Returns:
            Mã hành động
        """
        action_type = action_info.get("action_type", "hold")
        
        if self.action_type == "discrete":
            # Tìm mã hành động từ ánh xạ
            for action_idx, info in self.action_map.items():
                # So sánh action_type
                if info["action_type"] != action_type:
                    continue
                
                # So sánh chi tiết dựa trên action_type
                if action_type == "hold" or action_type == "close_all":
                    return action_idx
                
                elif action_type in ["buy", "sell"]:
                    if "size" in info and "size" in action_info:
                        if abs(info["size"] - action_info["size"]) < 0.01:
                            return action_idx
                
                elif action_type == "close":
                    if "position_id" in info and "position_id" in action_info:
                        if info["position_id"] == action_info["position_id"]:
                            return action_idx
            
            # Nếu không tìm thấy, trả về hành động hold
            return 0
        
        elif self.action_type == "multi_discrete":
            # Mã hóa hành động đa rời rạc
            if action_type == "hold":
                action = [0]
            
            elif action_type == "buy":
                action = [1]
                
                if self.position_sizing and "size" in action_info:
                    size = action_info["size"]
                    size_levels = [0.1, 0.25, 0.5, 0.75, 1.0]
                    size_idx = min(range(len(size_levels)), key=lambda i: abs(size_levels[i] - size))
                    action.append(size_idx)
            
            elif action_type == "sell" and self.allow_short:
                action = [2]
                
                if self.position_sizing and "size" in action_info:
                    size = action_info["size"]
                    size_levels = [0.1, 0.25, 0.5, 0.75, 1.0]
                    size_idx = min(range(len(size_levels)), key=lambda i: abs(size_levels[i] - size))
                    action.append(size_idx)
            
            elif action_type in ["close", "close_all"]:
                action = [2 if not self.allow_short else 3]
                
                if self.include_position_actions:
                    if action_type == "close_all":
                        action.append(0)
                    elif "position_id" in action_info:
                        pos_idx = action_info["position_id"]
                        action.append(pos_idx + 1)
                    else:
                        action.append(0)
            
            else:
                # Mặc định
                action = [0]
            
            # Đảm bảo đầy đủ phần tử
            while len(action) < len(self.action_space.nvec):
                action.append(0)
            
            return np.array(action)
        
        elif self.action_type == "continuous":
            # Mã hóa hành động liên tục
            if action_type == "hold":
                action = [0.25]  # Giữa 0 và 0.5
            
            elif action_type == "buy":
                action = [0.75]  # Giữa 0.5 và 1.0
                
                if self.position_sizing and "size" in action_info:
                    size = action_info["size"]
                    # Ánh xạ size từ [0.1, 1.0] sang [-1, 1]
                    size_value = (size - 0.1) / 0.9 * 2 - 1
                    action.append(size_value)
                else:
                    action.append(1.0)  # Kích thước tối đa
            
            elif action_type == "sell" and self.allow_short:
                action = [-0.75]  # Giữa -1.0 và -0.5
                
                if self.position_sizing and "size" in action_info:
                    size = action_info["size"]
                    # Ánh xạ size từ [0.1, 1.0] sang [-1, 1]
                    size_value = (size - 0.1) / 0.9 * 2 - 1
                    action.append(size_value)
                else:
                    action.append(1.0)  # Kích thước tối đa
            
            elif action_type in ["close", "close_all"]:
                action = [-0.25]  # Giữa -0.5 và 0
                
                if self.include_position_actions and action_type == "close" and "position_id" in action_info:
                    pos_idx = action_info["position_id"]
                    # Ánh xạ pos_idx từ [0, max_positions-1] sang [-1, 1]
                    pos_value = pos_idx / (self.max_positions - 1) * 2 - 1 if self.max_positions > 1 else 0
                    
                    if len(action) > 1:
                        action.append(1.0)  # Đảm bảo có đủ phần tử
                    
                    action.append(pos_value)
                elif len(action) > 1:
                    action.append(0.0)  # Giá trị mặc định
                
                if len(action) > 2:
                    action.append(-1.0)  # Giá trị mặc định
            
            else:
                # Mặc định
                action = [0.0]
            
            # Đảm bảo đầy đủ phần tử
            while len(action) < self.action_space.shape[0]:
                action.append(0.0)
            
            return np.array(action)
        
        else:
            self.logger.error(f"Loại hành động không được hỗ trợ: {self.action_type}")
            return 0
    
    def sample(self) -> Any:
        """
        Lấy mẫu một hành động ngẫu nhiên.
        
        Returns:
            Mã hành động ngẫu nhiên
        """
        return self.action_space.sample()
    
    def get_action_info(self, action: Any = None) -> List[Dict[str, Any]]:
        """
        Lấy thông tin về tất cả các hành động có thể.
        
        Args:
            action: Mã hành động cụ thể để lấy thông tin (None để lấy tất cả)
            
        Returns:
            Danh sách các thông tin hành động
        """
        if self.action_type == "discrete":
            if action is not None:
                return [self.action_map.get(action, {"action_type": "unknown"})]
            else:
                return list(self.action_map.values())
        
        elif action is not None:
            return [self.decode_action(action)]
        
        # Trường hợp khác, tạo tất cả các hành động có thể
        action_infos = []
        
        # Hành động hold
        action_infos.append({"action_type": "hold"})
        
        # Hành động mua
        if self.position_sizing:
            size_levels = [0.1, 0.25, 0.5, 0.75, 1.0]
            for size in size_levels:
                action_infos.append({"action_type": "buy", "size": size})
        else:
            action_infos.append({"action_type": "buy", "size": 1.0})
        
        # Hành động bán (short)
        if self.allow_short:
            if self.position_sizing:
                for size in size_levels:
                    action_infos.append({"action_type": "sell", "size": size})
            else:
                action_infos.append({"action_type": "sell", "size": 1.0})
        
        # Hành động đóng tất cả
        action_infos.append({"action_type": "close_all"})
        
        # Hành động đóng vị thế cụ thể
        if self.include_position_actions:
            for pos_idx in range(self.max_positions):
                action_infos.append({"action_type": "close", "position_id": pos_idx})
        
        return action_infos