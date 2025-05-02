"""
Agent PPO (Proximal Policy Optimization).
File này định nghĩa lớp PPOAgent, thực hiện thuật toán PPO để
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

class PPOAgent(BaseAgent):
    """
    Agent sử dụng thuật toán Proximal Policy Optimization (PPO).
    PPO cải thiện độ ổn định của quá trình huấn luyện bằng cách giới hạn
    sự thay đổi của chính sách trong mỗi bước cập nhật.
    """
    
    def __init__(
        self,
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: int,
        action_type: str = "discrete",
        action_bound: Optional[Tuple[float, float]] = None,
        hidden_layers: List[int] = [64, 64],
        activation: str = "relu",
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        train_epochs: int = 10,
        batch_size: int = 64,
        use_shared_network: bool = True,
        normalize_advantages: bool = True,
        normalize_rewards: bool = False,
        use_gae: bool = True,
        target_kl: Optional[float] = 0.01,
        model_dir: Optional[str] = None,
        name: str = "ppo_agent",
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Khởi tạo PPO Agent.
        
        Args:
            state_dim: Kích thước không gian trạng thái
            action_dim: Kích thước không gian hành động
            action_type: Loại hành động ('discrete' hoặc 'continuous')
            action_bound: Giới hạn hành động cho continuous actions (min, max)
            hidden_layers: Danh sách số lượng neuron trong các lớp ẩn
            activation: Hàm kích hoạt
            learning_rate: Tốc độ học
            gamma: Hệ số chiết khấu
            lambda_gae: Hệ số lambda cho Generalized Advantage Estimation
            clip_ratio: Tỷ lệ cắt cho PPO
            entropy_coef: Hệ số entropy
            value_coef: Hệ số loss của value function
            max_grad_norm: Giới hạn norm gradient
            train_epochs: Số epochs trong mỗi lần cập nhật
            batch_size: Kích thước batch
            use_shared_network: Sử dụng mạng chia sẻ hay không
            normalize_advantages: Chuẩn hóa advantages hay không
            normalize_rewards: Chuẩn hóa rewards hay không
            use_gae: Sử dụng Generalized Advantage Estimation hay không
            target_kl: Ngưỡng KL divergence để dừng huấn luyện sớm
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
        
        # Thiết lập các thuộc tính bổ sung
        self.action_type = action_type
        self.action_bound = action_bound
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.use_shared_network = use_shared_network
        self.normalize_advantages = normalize_advantages
        self.normalize_rewards = normalize_rewards
        self.use_gae = use_gae
        self.target_kl = target_kl
        
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
            'values': [],
            'log_probs': [],
            'policies': []
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
            policy = self.policy_network.predict_policy(state)
        
        # Tính log probability của hành động
        log_prob = self._compute_log_prob(policy, action)
        
        # Thông tin bổ sung
        info = {
            'value': value,
            'log_prob': log_prob,
            'policy': policy
        }
        
        return action, info
    
    def _compute_log_prob(self, policy: Union[np.ndarray, List[np.ndarray]], action: Union[int, np.ndarray]) -> float:
        """
        Tính log probability của hành động.
        
        Args:
            policy: Policy output từ mạng (logits cho discrete, [mean, std] cho continuous)
            action: Hành động đã chọn
            
        Returns:
            Log probability của hành động
        """
        if self.action_type == 'discrete':
            # Chuyển logits thành probabilities
            if isinstance(policy, list):
                policy = policy[0]  # Nếu policy là list, lấy phần tử đầu tiên (logits)
            
            probs = tf.nn.softmax(policy).numpy()
            
            # Lấy probability của hành động đã chọn
            if isinstance(action, np.ndarray) and action.size == 1:
                action = action.item()
            
            prob = probs[0, action] if len(probs.shape) > 1 else probs[action]
            
            # Tính log probability
            return np.log(prob + 1e-10)
        else:
            # Đối với continuous action
            mean, std = policy
            
            # Normalize action về phạm vi [-1, 1]
            low, high = self.action_bound
            normalized_action = 2.0 * (action - low) / (high - low) - 1.0
            
            # Tính log probability sử dụng phân phối normal
            # log_prob = -0.5 * (((normalized_action - mean) / (std + 1e-10)) ** 2 + 2 * np.log(std + 1e-10) + np.log(2 * np.pi))
            # Tổng log probability theo tất cả các chiều
            # return log_prob.sum()
            
            # Sử dụng TensorFlow để tính log probability
            action_tensor = tf.convert_to_tensor(normalized_action, dtype=tf.float32)
            mean_tensor = tf.convert_to_tensor(mean[0] if len(mean.shape) > 1 else mean, dtype=tf.float32)
            std_tensor = tf.convert_to_tensor(std[0] if len(std.shape) > 1 else std, dtype=tf.float32)
            
            dist = tfp.distributions.Normal(mean_tensor, std_tensor)
            log_prob = dist.log_prob(action_tensor)
            
            return tf.reduce_sum(log_prob).numpy()
    
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
        self.memory['log_probs'].append(info['log_prob'])
        self.memory['policies'].append(info['policy'])
    
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
        old_log_probs = np.array(self.memory['log_probs'])
        old_policies = self.memory['policies']
        
        # Tính returns và advantages
        returns, advantages = self._compute_returns_and_advantages(rewards, values, dones)
        
        # Chuẩn hóa advantages
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Số mẫu và indices
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        # Thông tin huấn luyện
        train_info = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
            'explained_variance': 0.0,
            'stop_iteration': 0
        }
        
        # Huấn luyện trong nhiều epochs
        for epoch in range(self.train_epochs):
            # Shuffle dữ liệu
            np.random.shuffle(indices)
            
            # Huấn luyện theo batch
            epoch_info = {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy_loss': 0.0,
                'total_loss': 0.0
            }
            
            # Theo dõi KL divergence
            approx_kls = []
            
            for start_idx in range(0, n_samples, self.batch_size):
                # Lấy batch
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Tạo batch old_policies (có thể là list hoặc mảng numpy)
                if isinstance(old_policies[0], np.ndarray):
                    batch_old_policies = np.array([old_policies[i] for i in batch_indices])
                else:
                    # Nếu là list của 2 arrays (mean, std)
                    batch_old_means = np.array([old_policies[i][0] for i in batch_indices])
                    batch_old_stds = np.array([old_policies[i][1] for i in batch_indices])
                    batch_old_policies = [batch_old_means, batch_old_stds]
                
                # Huấn luyện trên batch
                if self.use_shared_network:
                    # Huấn luyện mạng chia sẻ
                    batch_info = self.network.train_on_batch(
                        batch_states, batch_actions, batch_advantages, batch_returns, 
                        batch_old_policies, self.clip_ratio
                    )
                else:
                    # Huấn luyện mạng policy riêng biệt
                    policy_info = self.policy_network.train_on_batch(
                        batch_states, batch_actions, batch_advantages, 
                        batch_old_policies, self.clip_ratio
                    )
                    
                    # Huấn luyện mạng value riêng biệt
                    value_loss = self.value_network.train_on_batch(batch_states, batch_returns)
                    
                    # Kết hợp thông tin
                    batch_info = {**policy_info, 'value_loss': value_loss}
                
                # Cập nhật thông tin epoch
                for key in epoch_info:
                    if key in batch_info:
                        epoch_info[key] += batch_info[key] * (end_idx - start_idx) / n_samples
                
                # Tính xấp xỉ KL divergence
                if 'mean_ratio' in batch_info:
                    # Xấp xỉ KL từ ratio
                    ratio = np.exp(batch_info['mean_ratio'])
                    approx_kl = np.mean((ratio - 1) - np.log(ratio))
                    approx_kls.append(approx_kl)
            
            # Cập nhật thông tin huấn luyện
            for key in epoch_info:
                train_info[key] = epoch_info[key]
            
            # Tính trung bình KL divergence
            if approx_kls:
                train_info['approx_kl'] = np.mean(approx_kls)
                
                # Kiểm tra early stopping dựa trên KL divergence
                if self.target_kl is not None and train_info['approx_kl'] > 1.5 * self.target_kl:
                    self.logger.info(f"Early stopping at epoch {epoch} due to reaching max kl: {train_info['approx_kl']:.4f}")
                    train_info['stop_iteration'] = epoch
                    break
        
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
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính returns và advantages.
        
        Args:
            rewards: Mảng phần thưởng
            values: Mảng giá trị trạng thái
            dones: Mảng đánh dấu kết thúc episode
            
        Returns:
            Tuple (returns, advantages)
        """
        n_steps = len(rewards)
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        if self.use_gae:
            # Generalized Advantage Estimation
            last_gae_lambda = 0
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    next_non_terminal = 1.0 - dones[t]
                    if self.use_shared_network:
                        next_value = self.network.predict(np.expand_dims(self.memory['next_states'][t], axis=0))[1][0][0]
                    else:
                        next_value = self.value_network.predict(np.expand_dims(self.memory['next_states'][t], axis=0))[0][0]
                else:
                    next_non_terminal = 1.0 - dones[t+1]
                    next_value = values[t+1]
                
                delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
                last_gae_lambda = delta + self.gamma * self.lambda_gae * next_non_terminal * last_gae_lambda
                advantages[t] = last_gae_lambda
            
            # Returns = Advantages + Values
            returns = advantages + values
        else:
            # Compute returns using discounted rewards
            last_return = 0
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    next_non_terminal = 1.0 - dones[t]
                    if self.use_shared_network:
                        next_value = self.network.predict(np.expand_dims(self.memory['next_states'][t], axis=0))[1][0][0]
                    else:
                        next_value = self.value_network.predict(np.expand_dims(self.memory['next_states'][t], axis=0))[0][0]
                    last_return = rewards[t] + self.gamma * next_value * next_non_terminal
                else:
                    next_non_terminal = 1.0 - dones[t+1]
                    last_return = rewards[t] + self.gamma * last_return * next_non_terminal
                
                returns[t] = last_return
            
            # Compute advantages
            advantages = returns - values
        
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
            'lambda_gae': self.lambda_gae,
            'clip_ratio': self.clip_ratio,
            'entropy_coef': self.entropy_coef,
            'value_coef': self.value_coef,
            'max_grad_norm': self.max_grad_norm,
            'train_epochs': self.train_epochs,
            'batch_size': self.batch_size,
            'use_shared_network': self.use_shared_network,
            'normalize_advantages': self.normalize_advantages,
            'normalize_rewards': self.normalize_rewards,
            'use_gae': self.use_gae,
            'target_kl': self.target_kl,
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
        summary += f"Clip ratio: {self.clip_ratio}\n"
        summary += f"Train epochs: {self.train_epochs}\n"
        summary += f"Batch size: {self.batch_size}\n"
        
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