"""
Agent PPO (Proximal Policy Optimization).
File này định nghĩa lớp PPOAgent, thực hiện thuật toán PPO để
tối ưu hóa chính sách trong các bài toán Reinforcement Learning.
"""

import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
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
        learning_rate: float = 0.00005,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_ratio: float = 0.1,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        train_epochs: int = 5,
        batch_size: int = 128,
        use_shared_network: bool = True,
        normalize_advantages: bool = True,
        normalize_rewards: bool = False,
        use_gae: bool = True,
        target_kl: Optional[float] = 0.05,
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
        # Lưu trữ kwargs
        self.kwargs = kwargs  # Di chuyển lên đây để tránh lỗi 'kwargs' không tồn tại
        
        # Đảm bảo action_dim là số dương
        if action_dim <= 0:
            raise ValueError(f"action_dim phải là số dương, nhưng nhận được {action_dim}")
        
        # Thiết lập action_bound mặc định cho continuous actions
        if action_type == 'continuous' and action_bound is None:
            action_bound = (-1.0, 1.0)
            
        # Gọi constructor của lớp cơ sở
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            save_dir=model_dir,
            name=name,
            logger=logger,
            **kwargs
        )
        
        # Thiết lập các thuộc tính bổ sung
        self.action_type = action_type
        self.action_bound = action_bound
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.lambda_gae = lambda_gae
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.train_epochs = train_epochs
        self.use_shared_network = use_shared_network
        self.normalize_advantages = normalize_advantages
        self.normalize_rewards = normalize_rewards
        self.use_gae = use_gae
        self.target_kl = target_kl
        
        # Các thuộc tính cho việc chuẩn hóa rewards
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        
        # Đặt giá trị mặc định cho các hyperparameter nếu không được cung cấp
        self.update_freq = kwargs.get('update_freq', 1)
        self.buffer_size = kwargs.get('buffer_size', 10000)
        
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
    
    def act(self, state: np.ndarray, explore: bool = True) -> Union[int, np.ndarray]:
        """
        Chọn hành động dựa trên trạng thái hiện tại.
        
        Args:
            state: Trạng thái hiện tại
            explore: True nếu agent nên khám phá, False nếu chỉ khai thác
            
        Returns:
            Hành động được chọn
        """
        try:
            action, _ = self.select_action(state, deterministic=not explore)
            
            # Đảm bảo action nằm trong giới hạn an toàn
            if self.action_type == 'discrete':
                # Đảm bảo action là số nguyên và nằm trong giới hạn
                try:
                    if isinstance(action, np.ndarray):
                        action = int(action.flat[0])
                    else:
                        action = int(action)
                except (TypeError, ValueError):
                    self.logger.warning(f"Không thể chuyển action '{action}' thành int, sử dụng 0")
                    action = 0
                
                # Đảm bảo nằm trong giới hạn [0, action_dim-1]
                action = min(max(action, 0), self.action_dim - 1)
            
            return action
        except Exception as e:
            self.logger.error(f"Lỗi trong phương thức act: {str(e)}")
            # Trả về hành động mặc định an toàn
            if self.action_type == 'discrete':
                return 0  # Hành động 0 thường an toàn
            else:
                return np.zeros(self.action_dim)  # Vector 0 thường an toàn
    
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
        
        try:
            if self.use_shared_network:
                # Sử dụng mạng chia sẻ
                policy, value, sampled_action = self.network.predict(state)
                
                # Xử lý value
                if hasattr(value, 'shape') and len(value.shape) > 1:
                    value = value[0][0]
                else:
                    value = value[0]
                
                if deterministic:
                    # Chọn hành động tốt nhất
                    if self.action_type == 'discrete':
                        # Đối với discrete action: argmax của logits
                        if hasattr(policy, 'shape') and len(policy.shape) > 1:
                            action = np.argmax(policy[0])
                        else:
                            action = np.argmax(policy)
                        
                        # Đảm bảo action nằm trong khoảng [0, action_dim-1]
                        action = min(max(int(action), 0), self.action_dim - 1)
                    else:
                        # Đối với continuous action: mean (không thêm noise)
                        action = policy[0]  # mean
                else:
                    # Hành động đã được lấy mẫu
                    if self.action_type == 'discrete':
                        # Đảm bảo action nằm trong khoảng [0, action_dim-1]
                        if isinstance(sampled_action, np.ndarray) and sampled_action.size == 1:
                            action = min(max(int(sampled_action.item()), 0), self.action_dim - 1)
                        else:
                            action = min(max(int(sampled_action[0]), 0), self.action_dim - 1)
                    else:
                        action = sampled_action[0]  # Lấy phần tử đầu tiên vì state chỉ có 1 mẫu
            else:
                # Sử dụng mạng riêng biệt
                if deterministic:
                    # Chọn hành động tốt nhất
                    if self.action_type == 'discrete':
                        # Đối với discrete action: argmax của logits
                        logits = self.policy_network.predict_policy(state)
                        if hasattr(logits, 'shape') and len(logits.shape) > 1:
                            action = np.argmax(logits[0])
                        else:
                            action = np.argmax(logits)
                        
                        # Đảm bảo action nằm trong khoảng [0, action_dim-1]
                        action = min(max(int(action), 0), self.action_dim - 1)
                    else:
                        # Đối với continuous action: mean (không thêm noise)
                        mean, _ = self.policy_network.predict_policy(state)
                        action = mean[0]  # mean
                else:
                    # Lấy mẫu hành động từ phân phối
                    action = self.policy_network.predict(state)[0]
                    
                    # Đảm bảo discrete action nằm trong giới hạn hợp lệ
                    if self.action_type == 'discrete':
                        if isinstance(action, np.ndarray) and action.size == 1:
                            action = min(max(int(action.item()), 0), self.action_dim - 1)
                        else:
                            action = min(max(int(action), 0), self.action_dim - 1)
                
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
        except Exception as e:
            self.logger.error(f"Lỗi trong phương thức select_action: {str(e)}")
            # Fallback: trả về hành động mặc định an toàn
            if self.action_type == 'discrete':
                safe_action = 0  # Hành động 0 được coi là an toàn
            else:
                safe_action = np.zeros(self.action_dim)  # Không di chuyển là an toàn
            
            # Thông tin mặc định
            info = {
                'value': 0.0,
                'log_prob': 0.0,
                'policy': None
            }
            
            return safe_action, info
    
    def _compute_log_prob(self, policy: Union[np.ndarray, List[np.ndarray]], action: Union[int, np.ndarray]) -> float:
        """
        Tính log probability của hành động.
        
        Args:
            policy: Policy output từ mạng (logits cho discrete, [mean, std] cho continuous)
            action: Hành động đã chọn
            
        Returns:
            Log probability của hành động
        """
        try:
            if self.action_type == 'discrete':
                # Chuyển logits thành probabilities
                if isinstance(policy, list):
                    policy = policy[0]  # Nếu policy là list, lấy phần tử đầu tiên (logits)
                
                # Đảm bảo policy có kích thước đúng [batch_size, action_dim]
                if hasattr(policy, 'shape'):
                    # Kiểm tra nếu policy có 3 chiều [batch_size, 1, action_dim]
                    if len(policy.shape) == 3:
                        # Reshape từ [batch_size, 1, action_dim] thành [batch_size, action_dim]
                        policy = np.reshape(policy, (policy.shape[0], policy.shape[2]))
                    
                    # Kiểm tra kích thước sau khi reshape
                    if len(policy.shape) > 1 and policy.shape[1] != self.action_dim:
                        self.logger.warning(f"Kích thước policy không đúng: {policy.shape}, action_dim: {self.action_dim}")
                        # Tạo vector zeros với kích thước đúng và đặt giá trị 1 ở vị trí 0
                        tmp_policy = np.zeros((policy.shape[0], self.action_dim))
                        tmp_policy[:, 0] = 1.0
                        probs = tmp_policy
                    elif len(policy.shape) > 1:
                        probs = tf.nn.softmax(policy).numpy()
                    else:
                        # Nếu policy là vector 1D, kiểm tra kích thước
                        if isinstance(policy, np.ndarray) and policy.size != self.action_dim:
                            self.logger.warning(f"Kích thước policy 1D không đúng: {policy.size}, action_dim: {self.action_dim}")
                            # Tạo vector zeros với kích thước đúng và đặt giá trị 1 ở vị trí 0
                            tmp_policy = np.zeros(self.action_dim)
                            tmp_policy[0] = 1.0
                            probs = np.expand_dims(tmp_policy, axis=0)
                        else:
                            # Nếu policy là vector 1D, thêm chiều batch
                            probs = tf.nn.softmax(np.expand_dims(policy, axis=0)).numpy()
                else:
                    # Xử lý trường hợp policy không có thuộc tính shape
                    self.logger.warning(f"Policy không có thuộc tính shape")
                    tmp_policy = np.zeros((1, self.action_dim))
                    tmp_policy[:, 0] = 1.0
                    probs = tmp_policy
                
                # Xử lý action
                if isinstance(action, np.ndarray):
                    if action.size == 1:
                        action = action.item()
                    else:
                        # Nếu action là mảng nhiều phần tử, lấy phần tử đầu tiên
                        self.logger.warning(f"Action là mảng nhiều phần tử: {action}, lấy phần tử đầu tiên")
                        action = action.flat[0]
                
                # Đảm bảo action là số nguyên và nằm trong giới hạn
                try:
                    action = int(action)
                except (TypeError, ValueError):
                    self.logger.warning(f"Không thể chuyển action thành số nguyên: {action}, sử dụng 0")
                    action = 0
                
                # Đảm bảo action nằm trong giới hạn
                action = min(max(action, 0), self.action_dim - 1)
                
                # Lấy probability của hành động đã chọn
                if len(probs.shape) > 1:
                    prob = probs[0, action]
                else:
                    prob = probs[action]
                
                # Tính log probability
                return float(np.log(prob + 1e-10))
            else:
                # Đối với continuous action
                try:
                    if isinstance(policy, list) and len(policy) == 2:
                        mean, std = policy
                    else:
                        self.logger.warning(f"Policy không đúng định dạng cho continuous action: {policy}")
                        # Fallback: tạo mean và std mặc định
                        mean = np.zeros(self.action_dim)
                        std = np.ones(self.action_dim)
                    
                    # Normalize action về phạm vi [-1, 1]
                    low, high = self.action_bound
                    
                    # Đảm bảo action có kích thước phù hợp
                    if isinstance(action, np.ndarray):
                        if action.size != self.action_dim:
                            self.logger.warning(f"Kích thước action không khớp với action_dim: {action.size} vs {self.action_dim}")
                            # Điều chỉnh kích thước action
                            if action.size < self.action_dim:
                                # Nếu thiếu, bổ sung thêm 0
                                padded_action = np.zeros(self.action_dim)
                                padded_action[:action.size] = action.flat[:action.size]
                                action = padded_action
                            else:
                                # Nếu thừa, cắt bớt
                                action = action.flat[:self.action_dim]
                    else:
                        # Nếu action là scalar, chuyển thành mảng 1 phần tử
                        action = np.array([action])
                        if action.size != self.action_dim:
                            padded_action = np.zeros(self.action_dim)
                            padded_action[0] = action[0]
                            action = padded_action
                    
                    # Normalize action
                    normalized_action = 2.0 * (action - low) / (high - low) - 1.0
                    
                    # Đảm bảo kích thước đúng cho mean và std
                    if hasattr(mean, 'shape') and len(mean.shape) > 1:
                        mean_tensor = tf.convert_to_tensor(mean[0], dtype=tf.float32)
                    else:
                        mean_tensor = tf.convert_to_tensor(mean, dtype=tf.float32)
                    
                    if hasattr(std, 'shape') and len(std.shape) > 1:
                        std_tensor = tf.convert_to_tensor(std[0], dtype=tf.float32)
                    else:
                        std_tensor = tf.convert_to_tensor(std, dtype=tf.float32)
                    
                    # Đảm bảo kích thước đúng cho action
                    action_tensor = tf.convert_to_tensor(normalized_action, dtype=tf.float32)
                    
                    # Tạo phân phối normal và tính log prob
                    dist = tfp.distributions.Normal(mean_tensor, std_tensor)
                    log_prob = dist.log_prob(action_tensor)
                    
                    return float(tf.reduce_sum(log_prob).numpy())
                except Exception as e:
                    self.logger.error(f"Lỗi khi tính log_prob cho continuous action: {str(e)}")
                    return 0.0
        except Exception as e:
            self.logger.error(f"Lỗi trong _compute_log_prob: {str(e)}")
            return 0.0
    
    def remember(self, state: np.ndarray, action: Union[int, np.ndarray], 
                reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Lưu trữ trải nghiệm vào bộ nhớ.
        
        Args:
            state: Trạng thái hiện tại
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            next_state: Trạng thái kế tiếp
            done: True nếu episode kết thúc, False nếu không
        """
        # Tính giá trị và thông tin bổ sung từ trạng thái hiện tại
        try:
            # Đảm bảo action là kiểu đúng và nằm trong giới hạn
            if self.action_type == 'discrete':
                # Xử lý cho discrete action
                if isinstance(action, np.ndarray):
                    if action.size == 1:
                        action = int(action.item())
                    else:
                        self.logger.warning(f"Action có kích thước không mong đợi: {action.shape}, lấy phần tử đầu tiên")
                        action = int(action.flat[0])
                else:
                    # Cố gắng chuyển đổi action thành int
                    try:
                        action = int(action)
                    except (TypeError, ValueError):
                        self.logger.warning(f"Không thể chuyển action thành số nguyên: {action}, sử dụng 0")
                        action = 0
                
                # Đảm bảo action nằm trong giới hạn
                action = min(max(action, 0), self.action_dim - 1)
            
            if self.use_shared_network:
                policy, value, _ = self.network.predict(np.expand_dims(state, axis=0))
                value = value[0][0] if hasattr(value, 'shape') and len(value.shape) > 1 else value[0]
            else:
                policy = self.policy_network.predict_policy(np.expand_dims(state, axis=0))
                value = self.value_network.predict(np.expand_dims(state, axis=0))[0][0]
            
            # Tính log probability của action
            log_prob = self._compute_log_prob(policy, action)
            
            # Chuẩn hóa reward nếu cần
            normalized_reward = reward
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
            
            # Lưu vào bộ nhớ
            self.memory['states'].append(state)
            self.memory['actions'].append(action)
            self.memory['rewards'].append(normalized_reward)
            self.memory['next_states'].append(next_state)
            self.memory['dones'].append(done)
            self.memory['values'].append(value)
            self.memory['log_probs'].append(log_prob)
            self.memory['policies'].append(policy)
        except Exception as e:
            self.logger.error(f"Lỗi trong phương thức remember: {str(e)}")
            # Vẫn lưu trạng thái cơ bản nếu có lỗi, đảm bảo action nằm trong giới hạn
            if self.action_type == 'discrete':
                # Đảm bảo action là số nguyên và nằm trong khoảng [0, action_dim-1]
                try:
                    if isinstance(action, np.ndarray):
                        action = int(action.flat[0])
                    else:
                        action = int(action)
                except (TypeError, ValueError):
                    action = 0
                action = min(max(action, 0), self.action_dim - 1)
            
            self.memory['states'].append(state)
            self.memory['actions'].append(action)
            self.memory['rewards'].append(reward)
            self.memory['next_states'].append(next_state)
            self.memory['dones'].append(done)
            # Thêm các giá trị mặc định cho các thuộc tính còn lại
            self.memory['values'].append(0.0)
            self.memory['log_probs'].append(0.0)
            # Tạo policy mặc định
            if self.action_type == 'discrete':
                default_policy = np.zeros(self.action_dim)
                default_policy[0] = 1.0  # Thiên vị hành động 0
            else:
                default_policy = [np.zeros(self.action_dim), np.ones(self.action_dim)]  # [mean, std]
            self.memory['policies'].append(default_policy)
    
    def learn(self) -> Dict[str, float]:
        """
        Học từ bộ nhớ kinh nghiệm.
        
        Returns:
            Dict chứa thông tin về quá trình học (loss, v.v.)
        """
        # Kiểm tra nếu bộ nhớ trống
        if len(self.memory['states']) == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0, 'total_loss': 0}
        
        if 'values' not in self.memory or len(self.memory['values']) == 0:
            # Tính lại values nếu cần
            self.memory['values'] = []
            for state in self.memory['states']:
                if self.use_shared_network:
                    _, value, _ = self.network.predict(np.expand_dims(state, axis=0))
                    self.memory['values'].append(value[0][0])
                else:
                    value = self.value_network.predict(np.expand_dims(state, axis=0))
                    self.memory['values'].append(value[0][0])
        
        if 'log_probs' not in self.memory or len(self.memory['log_probs']) == 0 or 'policies' not in self.memory or len(self.memory['policies']) == 0:
            # Tính lại log_probs và policies nếu cần
            self.memory['log_probs'] = []
            self.memory['policies'] = []
            for i, state in enumerate(self.memory['states']):
                action = self.memory['actions'][i]
                if self.use_shared_network:
                    policy, _, _ = self.network.predict(np.expand_dims(state, axis=0))
                else:
                    policy = self.policy_network.predict_policy(np.expand_dims(state, axis=0))
                
                log_prob = self._compute_log_prob(policy, action)
                
                self.memory['log_probs'].append(log_prob)
                self.memory['policies'].append(policy)
        
        # Gọi phương thức train để thực hiện huấn luyện
        try:
            return self.train()
        except Exception as e:
            self.logger.error(f"Lỗi trong phương thức learn: {str(e)}")
            return {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0, 'total_loss': 0}
    
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
        
        try:
            # Chuyển đổi dữ liệu thành mảng numpy
            states = np.array(self.memory['states'])
            rewards = np.array(self.memory['rewards'])
            next_states = np.array(self.memory['next_states'])
            dones = np.array(self.memory['dones'])
            values = np.array(self.memory['values'])
            old_log_probs = np.array(self.memory['log_probs'])
            
            # Xử lý actions đặc biệt để đảm bảo kích thước đúng
            memory_actions = self.memory['actions']
            if self.action_type == 'discrete':
                # Đảm bảo rằng actions là mảng 1D cho discrete actions và nằm trong giới hạn action_dim
                actions = []
                for act in memory_actions:
                    if isinstance(act, (int, np.integer, float, np.floating)):
                        # Đảm bảo giá trị action nằm trong giới hạn
                        act_val = min(max(0, int(act)), self.action_dim - 1)  # FIX: thêm max(0, ...) để đảm bảo không âm
                        actions.append(act_val)
                    elif isinstance(act, np.ndarray):
                        if act.size == 1:
                            act_val = min(int(act.item()), self.action_dim - 1)
                            actions.append(act_val)
                        else:
                            # Trường hợp hiếm: action là mảng nhiều chiều
                            act_val = min(max(0, int(act.flat[0])), self.action_dim - 1)
                            actions.append(act_val)
                    else:
                        # FIX: Sử dụng giá trị mặc định an toàn thay vì giữ nguyên
                        self.logger.warning(f"Loại action không xác định: {type(act)}, sử dụng giá trị mặc định 0")
                        actions.append(0)
                # Chuyển đổi thành mảng numpy
                actions = np.array(actions, dtype=np.int32)
            else:
                # Đối với continuous actions, cần xử lý cẩn thận hơn
                actions = np.array(memory_actions)
                
            # Xử lý policies cẩn thận
            old_policies = []
            for policy in self.memory['policies']:
                if isinstance(policy, list) and len(policy) == 2:
                    # Đây là policy cho continuous actions [mean, std]
                    old_policies.append(policy)
                else:
                    # Policy cho discrete actions hoặc cấu trúc khác
                    old_policies.append(policy)
            
            # Tính returns và advantages
            returns, advantages = self._compute_returns_and_advantages(rewards, values, dones)
            
            # Chuẩn hóa advantages
            if self.normalize_advantages:
                advantages = np.clip(advantages, -10.0, 10.0)
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
                    
                    # Tạo batch old_policies với cách xử lý cẩn thận hơn
                    if len(old_policies) == 0:
                        batch_old_policies = None
                        self.logger.warning("Không có old_policies để xử lý trong batch")
                    elif self.action_type == 'discrete':
                        try:
                            # Lấy các policies tương ứng với batch_indices
                            selected_policies = [old_policies[i] if i < len(old_policies) else old_policies[0] for i in batch_indices]
                            
                            # Kiểm tra xem các policies có kích thước giống nhau không
                            shapes = [np.array(p).shape for p in selected_policies if hasattr(p, 'shape') or hasattr(p, '__len__')]
                            
                            if len(shapes) > 0 and all(s == shapes[0] for s in shapes):
                                # Nếu tất cả có cùng kích thước, stack chúng
                                batch_old_policies = np.stack(selected_policies)
                            else:
                                # Nếu kích thước khác nhau, giữ dạng list
                                batch_old_policies = selected_policies
                                
                        except Exception as e:
                            self.logger.warning(f"Lỗi khi xử lý old_policies cho discrete actions: {str(e)}")
                            # Fallback an toàn
                            batch_old_policies = None
                    else:
                        # Xử lý cho continuous actions
                        try:
                            # Đảm bảo old_policies có định dạng đúng [mean, std]
                            if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in old_policies if p is not None):
                                # Lấy các policies tương ứng với batch_indices
                                selected_policies = [old_policies[i] if i < len(old_policies) else old_policies[0] for i in batch_indices]
                                
                                # Tách mean và std
                                means = [p[0] for p in selected_policies]
                                stds = [p[1] for p in selected_policies]
                                
                                # Stack means và stds riêng biệt
                                try:
                                    batch_old_means = np.stack(means)
                                    batch_old_stds = np.stack(stds)
                                    batch_old_policies = [batch_old_means, batch_old_stds]
                                except Exception as e:
                                    self.logger.warning(f"Không thể stack means/stds: {str(e)}")
                                    batch_old_policies = None
                            else:
                                self.logger.warning("Định dạng old_policies không đúng cho continuous actions")
                                batch_old_policies = None
                        except Exception as e:
                            self.logger.warning(f"Lỗi khi xử lý old_policies cho continuous actions: {str(e)}")
                            batch_old_policies = None
                    
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
                    if self.target_kl is not None and train_info['approx_kl'] > 2.0 * self.target_kl:
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
        except Exception as e:
            self.logger.error(f"Lỗi trong phương thức train: {str(e)}")
            # Xóa bộ nhớ nếu có lỗi để tránh huấn luyện lại trên dữ liệu có vấn đề
            self._clear_memory()
            return {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0, 'total_loss': 0}
    
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
                    try:
                        if self.use_shared_network:
                            _, value_output, _ = self.network.predict(np.expand_dims(self.memory['next_states'][t], axis=0))
                            if hasattr(value_output, 'shape') and len(value_output.shape) > 1:
                                next_value = value_output[0][0]
                            else:
                                next_value = float(value_output[0]) if isinstance(value_output, (list, np.ndarray)) else float(value_output)
                        else:
                            next_value = self.value_network.predict(np.expand_dims(self.memory['next_states'][t], axis=0))[0][0]
                    except Exception as e:
                        self.logger.warning(f"Lỗi khi dự đoán next_value: {str(e)}")
                        next_value = 0.0
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
                    try:
                        if self.use_shared_network:
                            _, value_output, _ = self.network.predict(np.expand_dims(self.memory['next_states'][t], axis=0))
                            if hasattr(value_output, 'shape') and len(value_output.shape) > 1:
                                next_value = value_output[0][0]
                            else:
                                next_value = float(value_output[0]) if isinstance(value_output, (list, np.ndarray)) else float(value_output)
                        else:
                            next_value = self.value_network.predict(np.expand_dims(self.memory['next_states'][t], axis=0))[0][0]
                    except Exception as e:
                        self.logger.warning(f"Lỗi khi dự đoán next_value: {str(e)}")
                        next_value = 0.0
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
    
    def _save_model_impl(self, path: Union[str, Path]) -> None:
        """
        Triển khai cụ thể của việc lưu mô hình.
        
        Args:
            path: Đường dẫn lưu mô hình
        """
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Lưu các mạng neural
        if self.use_shared_network:
            self.network.save(str(path))
        else:
            policy_path = Path(str(path) + "_policy")
            value_path = Path(str(path) + "_value")
            self.policy_network.save(str(policy_path))
            self.value_network.save(str(value_path))
        
        # Lưu các thông số khác như reward_mean, reward_std, v.v. nếu cần
        # Có thể sử dụng pickle hoặc json để lưu
        meta_data = {
            'reward_mean': float(self.reward_mean),
            'reward_std': float(self.reward_std),
            'reward_count': int(self.reward_count)
        }
        
        meta_path = Path(str(path) + "_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f)
    
    def _load_model_impl(self, path: Union[str, Path]) -> bool:
        """
        Triển khai cụ thể của việc tải mô hình.
        
        Args:
            path: Đường dẫn tải mô hình
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        path = Path(path)
        if not path.exists():
            self.logger.warning(f"Không tìm thấy mô hình tại {path}")
            return False
        
        try:
            # Tải các mạng neural
            if self.use_shared_network:
                success = self.network.load(str(path))
                if not success:
                    self.logger.error(f"Lỗi khi tải mạng chia sẻ từ {path}")
                    return False
            else:
                policy_path = Path(str(path) + "_policy")
                value_path = Path(str(path) + "_value")
                
                if not policy_path.exists() or not value_path.exists():
                    self.logger.warning(f"Không tìm thấy các file mô hình tại {policy_path} hoặc {value_path}")
                    return False
                
                policy_success = self.policy_network.load(str(policy_path))
                value_success = self.value_network.load(str(value_path))
                
                if not policy_success or not value_success:
                    self.logger.error(f"Lỗi khi tải mạng policy hoặc value")
                    return False
            
            # Tải các thông số khác
            meta_path = Path(str(path) + "_meta.json")
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                    self.reward_mean = meta_data.get('reward_mean', 0.0)
                    self.reward_std = meta_data.get('reward_std', 1.0)
                    self.reward_count = meta_data.get('reward_count', 0)
            
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tải mô hình: {str(e)}")
            return False
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Lưu mô hình.
        
        Args:
            filepath: Đường dẫn tới thư mục lưu mô hình (None để sử dụng thư mục mặc định)
            
        Returns:
            Đường dẫn đã lưu mô hình
        """
        if filepath is None:
            if self.save_dir is None:
                # Tạo thư mục mặc định dựa trên tên và thời gian hiện tại
                model_dir = f"saved_models/{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(model_dir, exist_ok=True)
                filepath = model_dir
            else:
                filepath = self.save_dir
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