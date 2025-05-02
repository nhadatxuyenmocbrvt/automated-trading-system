"""
Agent A2C (Advantage Actor-Critic).
File này định nghĩa lớp A2CAgent, thực hiện thuật toán A2C để
tối ưu hóa chính sách trong các bài toán Reinforcement Learning.
"""

import os
import time
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
from datetime import datetime
import json
from pathlib import Path

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from models.agents.base_agent import BaseAgent
from models.networks.policy_network import PolicyNetwork
from models.networks.value_network import ValueNetwork
from models.networks.shared_network import SharedNetwork

class A2CAgent(BaseAgent):
    """
    Agent sử dụng thuật toán Advantage Actor-Critic (A2C).
    A2C kết hợp actor (policy network) và critic (value network)
    để học đồng thời cả chính sách và giá trị trạng thái.
    """
    
    def __init__(
        self,
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: int,
        action_type: str = "discrete",
        action_bound: Optional[Tuple[float, float]] = None,
        hidden_layers: List[int] = [64, 64],
        activation: str = "relu",
        learning_rate: float = 0.0007,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_shared_network: bool = True,
        normalize_advantages: bool = True,
        normalize_rewards: bool = False,
        rms_prop_eps: float = 1e-5,
        model_dir: Optional[str] = None,
        name: str = "a2c_agent",
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Khởi tạo A2C Agent.
        
        Args:
            state_dim: Kích thước không gian trạng thái
            action_dim: Kích thước không gian hành động
            action_type: Loại hành động ('discrete' hoặc 'continuous')
            action_bound: Giới hạn hành động cho continuous actions (min, max)
            hidden_layers: Danh sách số lượng neuron trong các lớp ẩn
            activation: Hàm kích hoạt
            learning_rate: Tốc độ học
            gamma: Hệ số chiết khấu
            entropy_coef: Hệ số entropy để khuyến khích khám phá
            value_coef: Hệ số loss của value function
            max_grad_norm: Giới hạn norm gradient
            use_shared_network: Sử dụng mạng chia sẻ hay không
            normalize_advantages: Chuẩn hóa advantages hay không
            normalize_rewards: Chuẩn hóa rewards hay không
            rms_prop_eps: Epsilon cho RMSProp optimizer
            model_dir: Thư mục lưu mô hình
            name: Tên của agent
            logger: Logger tùy chỉnh
        """
        # Gọi constructor của lớp cơ sở
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            model_dir=model_dir,
            name=name,
            logger=logger,
            **kwargs
        )
        
        # Thiết lập các thuộc tính
        self.action_type = action_type
        self.action_bound = action_bound
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.use_shared_network = use_shared_network
        self.normalize_advantages = normalize_advantages
        self.normalize_rewards = normalize_rewards
        self.rms_prop_eps = rms_prop_eps
        
        # Các thuộc tính cho việc chuẩn hóa rewards
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        
        # Khởi tạo bộ nhớ
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'values': []
        }
        
        # Khởi tạo mạng neural
        self._build_networks()
        
        self.logger.info(
            f"Đã khởi tạo {self.__class__.__name__} với state_dim={state_dim}, "
            f"action_dim={action_dim}, action_type={action_type}, "
            f"use_shared_network={use_shared_network}"
        )
    
    def _build_networks(self) -> None:
        """
        Khởi tạo các mạng neural cho agent.
        """
        # Thiết lập optimizer
        optimizer_params = {
            'optimizer': 'rmsprop',
            'rms_prop_eps': self.rms_prop_eps
        }
        
        if self.use_shared_network:
            # Sử dụng mạng chia sẻ cho cả policy và value
            self.network = SharedNetwork(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_layers=self.hidden_layers,
                activation=self.activation,
                learning_rate=self.learning_rate,
                network_type=self.kwargs.get('network_type', 'mlp'),
                action_type=self.action_type,
                action_bound=self.action_bound,
                entropy_coef=self.entropy_coef,
                value_coef=self.value_coef,
                name=f"{self.name}_shared_network",
                logger=self.logger,
                **optimizer_params,
                **self.kwargs
            )
            
            # Set None để biết là không sử dụng mạng riêng biệt
            self.policy_network = None
            self.value_network = None
        else:
            # Sử dụng mạng riêng biệt cho policy và value
            self.policy_network = PolicyNetwork(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_layers=self.hidden_layers,
                activation=self.activation,
                learning_rate=self.learning_rate,
                network_type=self.kwargs.get('network_type', 'mlp'),
                action_type=self.action_type,
                action_bound=self.action_bound,
                entropy_coef=self.entropy_coef,
                name=f"{self.name}_policy_network",
                logger=self.logger,
                **optimizer_params,
                **self.kwargs
            )
            
            self.value_network = ValueNetwork(
                state_dim=self.state_dim,
                action_dim=0,  # Pure value network
                hidden_layers=self.hidden_layers,
                activation=self.activation,
                learning_rate=self.learning_rate,
                network_type=self.kwargs.get('network_type', 'mlp'),
                name=f"{self.name}_value_network",
                logger=self.logger,
                **optimizer_params,
                **self.kwargs
            )
            
            # Set None để biết là không sử dụng mạng chia sẻ
            self.network = None
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[Union[int, np.ndarray], Dict[str, Any]]:
        """
        Chọn hành động dựa trên trạng thái hiện tại.
        
        Args:
            state: Trạng thái hiện tại
            deterministic: Nếu True, chọn hành động tốt nhất; nếu False, lấy mẫu từ phân phối
            
        Returns:
            Tuple (action, info) với action là hành động được chọn và info là thông tin bổ sung
        """
        # Đảm bảo state có đúng kích thước
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
        
        if self.use_shared_network:
            # Sử dụng mạng chia sẻ
            policy, value, action = self.network.predict(state)
            value = value[0][0] if hasattr(value, 'shape') and len(value.shape) > 1 else value[0]
            
            if deterministic:
                # Chọn hành động tốt nhất
                if self.action_type == 'discrete':
                    # Đối với discrete action: argmax của logits
                    action = np.argmax(policy)
                else:
                    # Đối với continuous action: mean (không thêm noise)
                    action = policy[0]  # mean
            else:
                # Hành động đã được lấy mẫu bởi network.predict
                action = action[0]  # Lấy phần tử đầu tiên vì state chỉ có 1 mẫu
        else:
            # Sử dụng mạng riêng biệt
            if deterministic:
                # Chọn hành động tốt nhất
                if self.action_type == 'discrete':
                    # Đối với discrete action: argmax của logits
                    logits = self.policy_network.predict_policy(state)
                    action = np.argmax(logits)
                else:
                    # Đối với continuous action: mean (không thêm noise)
                    mean, _ = self.policy_network.predict_policy(state)
                    action = mean[0]  # mean
            else:
                # Lấy mẫu hành động từ phân phối
                action = self.policy_network.predict(state)[0]
            
            # Tính giá trị của trạng thái
            value = self.value_network.predict(state)[0][0]
        
        # Thông tin bổ sung
        info = {
            'value': value
        }
        
        return action, info
    
    def remember(
        self, 
        state: np.ndarray, 
        action: Union[int, np.ndarray], 
        reward: float, 
        next_state: np.ndarray, 
        done: bool, 
        info: Dict[str, Any]
    ) -> None:
        """
        Lưu một bước tương tác vào bộ nhớ.
        
        Args:
            state: Trạng thái
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            next_state: Trạng thái tiếp theo
            done: Đã kết thúc episode hay chưa
            info: Thông tin bổ sung
        """
        # Chuẩn hóa reward nếu cần
        if self.normalize_rewards:
            # Cập nhật thống kê running mean/std
            self.reward_count += 1
            delta = reward - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = reward - self.reward_mean
            self.reward_std += delta * delta2
            
            # Tính reward đã chuẩn hóa
            if self.reward_count > 1:
                std = np.sqrt(self.reward_std / self.reward_count)
                normalized_reward = (reward - self.reward_mean) / (std + 1e-8)
            else:
                normalized_reward = 0.0
            
            reward = normalized_reward
        
        # Lưu vào bộ nhớ
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['next_states'].append(next_state)
        self.memory['dones'].append(done)
        self.memory['values'].append(info['value'])
    
    def train(self) -> Dict[str, float]:
        """
        Huấn luyện agent trên dữ liệu đã thu thập.
        
        Returns:
            Dict chứa thông tin về quá trình huấn luyện
        """
        # Kiểm tra xem có đủ dữ liệu để huấn luyện không
        if len(self.memory['states']) == 0:
            self.logger.warning("Không đủ dữ liệu để huấn luyện")
            return {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0}
        
        # Chuyển đổi dữ liệu thành mảng numpy
        states = np.array(self.memory['states'])
        actions = np.array(self.memory['actions'])
        rewards = np.array(self.memory['rewards'])
        next_states = np.array(self.memory['next_states'])
        dones = np.array(self.memory['dones'])
        values = np.array(self.memory['values'])
        
        # Tính returns và advantages
        returns, advantages = self._compute_returns_and_advantages(rewards, values, dones, next_states)
        
        # Chuẩn hóa advantages
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Thông tin huấn luyện
        train_info = {}
        
        # Huấn luyện trên toàn bộ dữ liệu
        if self.use_shared_network:
            # Huấn luyện mạng chia sẻ
            train_info = self.network.train_on_batch(
                states, actions, advantages, returns
            )
        else:
            # Huấn luyện mạng policy riêng biệt
            policy_info = self.policy_network.train_on_batch(
                states, actions, advantages
            )
            
            # Huấn luyện mạng value riêng biệt
            value_loss = self.value_network.train_on_batch(states, returns)
            
            # Kết hợp thông tin
            train_info = {**policy_info, 'value_loss': value_loss}
        
        # Tính explained variance
        var_y = np.var(returns)
        explained_var = 1 - np.var(returns - values) / (var_y + 1e-8)
        train_info['explained_variance'] = explained_var
        
        # Xóa bộ nhớ sau khi huấn luyện
        self._clear_memory()
        
        return train_info
    
    def _compute_returns_and_advantages(
        self, 
        rewards: np.ndarray, 
        values: np.ndarray, 
        dones: np.ndarray,
        next_states: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính returns và advantages.
        
        Args:
            rewards: Mảng phần thưởng
            values: Mảng giá trị trạng thái
            dones: Mảng đánh dấu kết thúc episode
            next_states: Mảng trạng thái tiếp theo
            
        Returns:
            Tuple (returns, advantages)
        """
        n_steps = len(rewards)
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        # Tính giá trị next_state cuối cùng
        if n_steps > 0:
            last_state = next_states[-1]
            last_state = np.expand_dims(last_state, axis=0)
            
            if self.use_shared_network:
                _, last_value, _ = self.network.predict(last_state)
                last_value = last_value[0][0] if hasattr(last_value, 'shape') and len(last_value.shape) > 1 else last_value[0]
            else:
                last_value = self.value_network.predict(last_state)[0][0]
            
            # Kiểm tra xem episode đã kết thúc chưa
            if dones[-1]:
                last_value = 0.0
        else:
            last_value = 0.0
        
        # Tính n-step returns và advantages
        next_value = last_value
        for t in reversed(range(n_steps)):
            # Tính returns
            returns[t] = rewards[t] + self.gamma * next_value * (1.0 - dones[t])
            
            # Tính advantages
            # A(s,a) = R + γV(s') - V(s)
            advantages[t] = returns[t] - values[t]
            
            # Cập nhật next_value
            next_value = values[t]
        
        return returns, advantages
    
    def _clear_memory(self) -> None:
        """
        Xóa bộ nhớ sau khi đã huấn luyện.
        """
        for key in self.memory:
            self.memory[key] = []
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Lưu mô hình.
        
        Args:
            filepath: Đường dẫn tới thư mục lưu mô hình (None để sử dụng thư mục mặc định)
            
        Returns:
            Đường dẫn đã lưu mô hình
        """
        if filepath is None:
            if self.model_dir is None:
                # Tạo thư mục mặc định dựa trên tên và thời gian hiện tại
                model_dir = f"saved_models/{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(model_dir, exist_ok=True)
                filepath = model_dir
            else:
                filepath = self.model_dir
                os.makedirs(filepath, exist_ok=True)
        
        # Tạo đường dẫn đầy đủ
        path = Path(filepath)
        
        # Lưu các mạng neural
        if self.use_shared_network:
            self.network.save(str(path / "shared_network"))
        else:
            self.policy_network.save(str(path / "policy_network"))
            self.value_network.save(str(path / "value_network"))
        
        # Lưu cấu hình
        config = {
            'name': self.name,
            'state_dim': self.state_dim if isinstance(self.state_dim, int) else list(self.state_dim),
            'action_dim': self.action_dim,
            'action_type': self.action_type,
            'action_bound': self.action_bound,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'entropy_coef': self.entropy_coef,
            'value_coef': self.value_coef,
            'max_grad_norm': self.max_grad_norm,
            'use_shared_network': self.use_shared_network,
            'normalize_advantages': self.normalize_advantages,
            'normalize_rewards': self.normalize_rewards,
            'rms_prop_eps': self.rms_prop_eps,
            'reward_stats': {
                'mean': float(self.reward_mean),
                'std': float(self.reward_std),
                'count': int(self.reward_count)
            },
            'kwargs': self.kwargs
        }
        
        with open(path / "config.json", 'w') as f:
            json.dump(config, f, indent=4)
        
        self.logger.info(f"Đã lưu mô hình tại {path}")
        
        return str(path)
    
    def load(self, filepath: str) -> bool:
        """
        Tải mô hình.
        
        Args:
            filepath: Đường dẫn tới thư mục chứa mô hình
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        path = Path(filepath)
        
        # Kiểm tra xem thư mục có tồn tại không
        if not path.exists():
            self.logger.warning(f"Thư mục {path} không tồn tại")
            return False
        
        # Tải cấu hình
        try:
            with open(path / "config.json", 'r') as f:
                config = json.load(f)
            
            # Cập nhật các thuộc tính
            for key, value in config.items():
                if key != 'kwargs' and hasattr(self, key):
                    setattr(self, key, value)
            
            # Cập nhật reward stats
            if 'reward_stats' in config:
                self.reward_mean = config['reward_stats']['mean']
                self.reward_std = config['reward_stats']['std']
                self.reward_count = config['reward_stats']['count']
            
            # Xây dựng lại mạng neural
            self._build_networks()
            
            # Tải trọng số
            if self.use_shared_network:
                load_success = self.network.load(str(path / "shared_network"))
                if not load_success:
                    self.logger.error("Không thể tải mạng chia sẻ")
                    return False
            else:
                policy_success = self.policy_network.load(str(path / "policy_network"))
                value_success = self.value_network.load(str(path / "value_network"))
                
                if not (policy_success and value_success):
                    self.logger.error("Không thể tải mạng policy hoặc value")
                    return False
            
            self.logger.info(f"Đã tải mô hình từ {path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tải mô hình: {str(e)}")
            return False
    
    def summary(self) -> str:
        """
        Tạo tóm tắt thông tin về agent.
        
        Returns:
            Chuỗi tóm tắt
        """
        summary = f"=== {self.name} ===\n"
        summary += f"State dimension: {self.state_dim}\n"
        summary += f"Action dimension: {self.action_dim}\n"
        summary += f"Action type: {self.action_type}\n"
        summary += f"Use shared network: {self.use_shared_network}\n"
        summary += f"Learning rate: {self.learning_rate}\n"
        summary += f"Gamma: {self.gamma}\n"
        
        summary += "\n=== Networks ===\n"
        
        # Hiển thị thông tin mạng neural
        if self.use_shared_network:
            self.network.summary()
            summary += "Using shared network for policy and value\n"
        else:
            self.policy_network.summary()
            self.value_network.summary()
            summary += "Using separate networks for policy and value\n"
        
        return summary