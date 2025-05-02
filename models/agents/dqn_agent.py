"""
Agent DQN (Deep Q-Network).
File này định nghĩa lớp DQNAgent cho thuật toán Deep Q-Network và các biến thể,
được sử dụng để huấn luyện agent quyết định hành động dựa trên Q-learning.
"""

# Thư viện chuẩn
import sys
import os
import random
import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

# Thư viện bên thứ ba
import numpy as np
import tensorflow as tf

# Thêm thư mục gốc vào path để import được các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from environments.base_environment import BaseEnvironment
from models.agents.base_agent import BaseAgent
from models.networks.value_network import ValueNetwork
class DQNAgent(BaseAgent):
    """
    Agent thực hiện thuật toán Deep Q-Network.
    Hỗ trợ các tính năng: experience replay, target network, double DQN, dueling DQN.
    """
    
    def __init__(
        self,
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: int,
        env: Optional[BaseEnvironment] = None,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 10000,
        update_target_freq: int = 100,
        hidden_layers: List[int] = [64, 64],
        activation: str = 'relu',
        network_type: str = 'mlp',
        double_dqn: bool = False,
        dueling: bool = False,
        prioritized_replay: bool = False,
        per_alpha: float = 0.6,  # Ưu tiên sampling
        per_beta: float = 0.4,   # Trọng số Importance-Sampling
        per_beta_increment: float = 0.001,  # Tăng beta theo thời gian
        save_dir: Optional[Union[str, Path]] = None,
        name: str = "dqn_agent",
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Khởi tạo agent DQN.
        
        Args:
            state_dim: Kích thước không gian trạng thái
            action_dim: Kích thước không gian hành động
            env: Môi trường tương tác
            learning_rate: Tốc độ học
            gamma: Hệ số giảm phần thưởng
            epsilon: Tham số khám phá ban đầu (cho ε-greedy)
            epsilon_min: Giá trị epsilon tối thiểu
            epsilon_decay: Tốc độ giảm epsilon sau mỗi bước
            batch_size: Kích thước batch huấn luyện
            memory_size: Kích thước bộ nhớ kinh nghiệm
            update_target_freq: Tần suất cập nhật mạng mục tiêu
            hidden_layers: Danh sách số lượng neuron trong các lớp ẩn
            activation: Hàm kích hoạt
            network_type: Loại mạng neural ('mlp', 'cnn', 'rnn')
            double_dqn: Sử dụng Double DQN hay không
            dueling: Sử dụng Dueling DQN hay không
            prioritized_replay: Sử dụng Prioritized Experience Replay hay không
            per_alpha: Hệ số alpha cho prioritized replay (độ ưu tiên)
            per_beta: Hệ số beta cho prioritized replay (trọng số IS)
            per_beta_increment: Tốc độ tăng beta
            save_dir: Thư mục lưu mô hình
            name: Tên của agent
            logger: Logger tùy chỉnh
        """
        # Gọi constructor của lớp cơ sở
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            env=env,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size,
            memory_size=memory_size,
            update_target_freq=update_target_freq,
            save_dir=save_dir,
            name=name,
            logger=logger,
            hidden_layers=hidden_layers,
            activation=activation,
            network_type=network_type,
            double_dqn=double_dqn,
            dueling=dueling,
            prioritized_replay=prioritized_replay,
            per_alpha=per_alpha,
            per_beta=per_beta,
            per_beta_increment=per_beta_increment,
            **kwargs
        )
        
        # Lưu các tham số bổ sung
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.network_type = network_type
        self.double_dqn = double_dqn
        self.dueling = dueling
        self.prioritized_replay = prioritized_replay
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_increment = per_beta_increment
        
        # Khởi tạo mạng Q chính và mạng Q mục tiêu
        self.q_network = ValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            learning_rate=learning_rate,
            network_type=network_type,
            dueling=dueling,
            name=f"{name}_q_network",
            logger=logger,
            **kwargs
        )
        
        self.target_q_network = ValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            learning_rate=learning_rate,
            network_type=network_type,
            dueling=dueling,
            name=f"{name}_target_q_network",
            logger=logger,
            **kwargs
        )
        
        # Sao chép trọng số từ Q-network sang target Q-network
        self.update_target_network()
        
        # Khởi tạo bộ nhớ
        if self.prioritized_replay:
            # Prioritized Experience Replay
            self.memory = []
            self.priorities = np.zeros(memory_size, dtype=np.float32)
            self.memory_counter = 0
        else:
            # Uniform Experience Replay
            self.memory = deque(maxlen=memory_size)
        
        self.logger.info(
            f"Đã khởi tạo {self.__class__.__name__} với state_dim={state_dim}, "
            f"action_dim={action_dim}, double_dqn={double_dqn}, dueling={dueling}, "
            f"prioritized_replay={prioritized_replay}"
        )
    
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Chọn hành động dựa trên trạng thái hiện tại.
        
        Args:
            state: Trạng thái hiện tại
            explore: True nếu agent nên khám phá, False nếu chỉ khai thác
            
        Returns:
            Hành động được chọn
        """
        # Khám phá (exploration)
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        # Khai thác (exploitation)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])  # Chọn hành động có Q-value cao nhất
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """
        Lưu trữ trải nghiệm vào bộ nhớ.
        
        Args:
            state: Trạng thái hiện tại
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            next_state: Trạng thái kế tiếp
            done: True nếu episode kết thúc, False nếu không
        """
        experience = (state, action, reward, next_state, done)
        
        if self.prioritized_replay:
            # Đối với Prioritized Experience Replay
            # Ban đầu, gán ưu tiên cao nhất cho trải nghiệm mới
            max_priority = np.max(self.priorities) if self.memory_counter > 0 else 1.0
            
            if len(self.memory) < self.memory_size:
                self.memory.append(experience)
            else:
                # Ghi đè index cũ nhất
                self.memory[self.memory_counter % self.memory_size] = experience
            
            # Cập nhật ưu tiên
            self.priorities[self.memory_counter % self.memory_size] = max_priority
            self.memory_counter += 1
        else:
            # Đối với Uniform Experience Replay
            self.memory.append(experience)
    
    def learn(self) -> Dict[str, float]:
        """
        Học từ bộ nhớ kinh nghiệm.
        
        Returns:
            Dict chứa thông tin về quá trình học (loss, v.v.)
        """
        # Kiểm tra xem có đủ mẫu trong bộ nhớ không
        if self.prioritized_replay:
            min_samples = min(len(self.memory), self.batch_size)
        else:
            min_samples = min(len(self.memory), self.batch_size)
        
        if min_samples < self.batch_size:
            return {}  # Không đủ mẫu để học
        
        # Lấy batch từ bộ nhớ
        if self.prioritized_replay:
            # Prioritized Experience Replay sampling
            batch, indices, is_weights = self._sample_prioritized_batch()
        else:
            # Uniform sampling
            batch = random.sample(self.memory, self.batch_size)
            indices = None
            is_weights = None
        
        # Chuẩn bị dữ liệu batch
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        # Tính Q-target
        if self.double_dqn:
            # Double DQN: Sử dụng Q-network để chọn hành động và target Q-network để đánh giá
            q_next = self.q_network.predict(next_states)
            q_target_next = self.target_q_network.predict(next_states)
            
            # Chọn hành động từ Q-network
            best_actions = np.argmax(q_next, axis=1)
            
            # Đánh giá bằng target Q-network
            q_target = rewards + (1 - dones) * self.gamma * q_target_next[np.arange(self.batch_size), best_actions]
        else:
            # Vanilla DQN
            q_target_next = self.target_q_network.predict(next_states)
            q_target = rewards + (1 - dones) * self.gamma * np.max(q_target_next, axis=1)
        
        # Huấn luyện Q-network
        # Lấy Q-values hiện tại
        q_values = self.q_network.predict(states)
        
        # Cập nhật Q-values cho hành động đã thực hiện
        target_f = q_values.copy()
        for i, action in enumerate(actions):
            target_f[i, action] = q_target[i]
        
        # Huấn luyện với trọng số importance sampling nếu sử dụng PER
        if self.prioritized_replay:
            # Áp dụng trọng số IS
            loss = self.q_network.model.train_on_batch(states, target_f, sample_weight=is_weights)
            
            # Cập nhật ưu tiên dựa trên TD-error
            td_errors = np.abs(q_target - q_values[np.arange(self.batch_size), actions])
            for i, idx in enumerate(indices):
                self.priorities[idx] = td_errors[i] ** self.per_alpha
            
            # Tăng beta
            self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)
        else:
            # Huấn luyện bình thường
            loss = self.q_network.model.train_on_batch(states, target_f)
        
        # Cập nhật target network nếu đến thời điểm
        self.train_count += 1
        if self.train_count % self.update_target_freq == 0:
            self.update_target_network()
        
        return {'loss': loss}
    
    def update_target_network(self) -> None:
        """
        Cập nhật trọng số từ Q-network sang target Q-network.
        """
        self.target_q_network.set_weights(self.q_network.get_weights())
        self.logger.debug(f"Đã cập nhật target network sau {self.train_count} bước huấn luyện")
    
    def _sample_prioritized_batch(self) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Lấy mẫu batch dựa trên ưu tiên (cho Prioritized Experience Replay).
        
        Returns:
            Tuple (batch, indices, importance_sampling_weights)
        """
        memory_size = min(self.memory_counter, self.memory_size)
        
        # Tính xác suất lấy mẫu dựa trên ưu tiên
        priorities = self.priorities[:memory_size]
        probabilities = priorities / np.sum(priorities)
        
        # Lấy mẫu indices
        indices = np.random.choice(memory_size, self.batch_size, replace=False, p=probabilities)
        
        # Lấy experiences từ indices
        batch = [self.memory[idx] for idx in indices]
        
        # Tính trọng số IS (Importance-Sampling)
        is_weights = np.power(memory_size * probabilities[indices], -self.per_beta)
        is_weights = is_weights / np.max(is_weights)  # Normalize
        
        return batch, indices, is_weights
    
    def _save_model_impl(self, path: Union[str, Path]) -> None:
        """
        Triển khai cụ thể của việc lưu mô hình.
        
        Args:
            path: Đường dẫn lưu mô hình
        """
        # Tạo thư mục nếu chưa tồn tại
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Lưu Q-network
        q_network_path = path.parent / f"{path.stem}_q_network{path.suffix}"
        self.q_network.save(str(q_network_path))
        
        # Lưu target Q-network
        target_path = path.parent / f"{path.stem}_target{path.suffix}"
        self.target_q_network.save(str(target_path))
    
    def _load_model_impl(self, path: Union[str, Path]) -> bool:
        """
        Triển khai cụ thể của việc tải mô hình.
        
        Args:
            path: Đường dẫn tải mô hình
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        path = Path(path)
        
        # Kiểm tra tính khả dụng của file
        q_network_path = path.parent / f"{path.stem}_q_network{path.suffix}"
        target_path = path.parent / f"{path.stem}_target{path.suffix}"
        
        if not q_network_path.exists() or not target_path.exists():
            self.logger.warning(f"Không tìm thấy các file mô hình cần thiết tại {path.parent}")
            return False
        
        # Tải Q-network
        q_network_loaded = self.q_network.load(str(q_network_path))
        
        # Tải target Q-network
        target_loaded = self.target_q_network.load(str(target_path))
        
        return q_network_loaded and target_loaded