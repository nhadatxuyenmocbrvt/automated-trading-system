"""
Lớp agent tổng hợp (Ensemble Agent).
File này định nghĩa lớp EnsembleAgent kế thừa từ BaseAgent,
kết hợp nhiều agent thành một "super agent", sử dụng các kỹ thuật tổng hợp
như voting, weighted average, stacking, và bagging.
"""

import os
import time
import logging
import numpy as np
import json
import random
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
import tensorflow as tf
from enum import Enum

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from models.agents.base_agent import BaseAgent
from environments.base_environment import BaseEnvironment
from agent_manager.agent_coordinator import AgentCombinationMethod

class EnsembleMethod(Enum):
    """
    Phương pháp tổng hợp agent.
    Định nghĩa các kỹ thuật để kết hợp nhiều agent.
    """
    VOTING = "voting"                # Bỏ phiếu (majority, weighted)
    AVERAGING = "averaging"          # Trung bình hóa hành động
    BAGGING = "bagging"              # Bootstrap Aggregating
    STACKING = "stacking"            # Stacked Generalization
    SWITCHING = "switching"          # Chuyển đổi dynamic giữa các agent
    CASCADING = "cascading"          # Cascade các agent theo thứ tự
    BOOSTING = "boosting"            # Boosting (AdaBoost-like)
    CUSTOM = "custom"                # Phương pháp tùy chỉnh

class EnsembleAgent(BaseAgent):
    """
    Lớp agent tổng hợp.
    Tổng hợp nhiều agent thành một "super agent" sử dụng
    các kỹ thuật tổng hợp nâng cao.
    """
    
    def __init__(
        self,
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: int,
        agents: List[BaseAgent],
        ensemble_method: Union[str, EnsembleMethod] = EnsembleMethod.VOTING,
        voting_method: Union[str, AgentCombinationMethod] = AgentCombinationMethod.WEIGHTED_VOTE,
        env: Optional[BaseEnvironment] = None,
        meta_model_path: Optional[Union[str, Path]] = None,
        name: str = "ensemble_agent",
        logger: Optional[logging.Logger] = None,
        save_dir: Optional[Union[str, Path]] = None,
        learning_rate: float = 0.001,
        performance_window: int = 100,
        **kwargs
    ):
        """
        Khởi tạo agent tổng hợp.
        
        Args:
            state_dim: Kích thước không gian trạng thái
            action_dim: Kích thước không gian hành động
            agents: Danh sách các agent cơ sở để tổng hợp
            ensemble_method: Phương pháp tổng hợp
            voting_method: Phương pháp bỏ phiếu (khi ensemble_method là VOTING)
            env: Môi trường tương tác
            meta_model_path: Đường dẫn đến meta-model (cho stacking)
            name: Tên của agent
            logger: Logger tùy chỉnh
            save_dir: Thư mục lưu trữ
            learning_rate: Tốc độ học cho meta-model
            performance_window: Kích thước cửa sổ theo dõi hiệu suất
            **kwargs: Các tham số tùy chọn khác
        """
        # Kiểm tra danh sách agent
        if not agents:
            raise ValueError("Danh sách agent không được để trống")
        
        # Khởi tạo lớp cha (BaseAgent)
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            env=env,
            learning_rate=learning_rate,
            name=name,
            logger=logger,
            save_dir=save_dir,
            **kwargs
        )
        
        # Thiết lập danh sách agent
        self.agents = agents
        
        # Thiết lập phương pháp tổng hợp
        if isinstance(ensemble_method, str):
            try:
                self.ensemble_method = EnsembleMethod(ensemble_method)
            except ValueError:
                self.logger.warning(f"Phương pháp tổng hợp '{ensemble_method}' không hợp lệ, sử dụng 'voting'")
                self.ensemble_method = EnsembleMethod.VOTING
        else:
            self.ensemble_method = ensemble_method
        
        # Thiết lập phương pháp bỏ phiếu
        if isinstance(voting_method, str):
            try:
                self.voting_method = AgentCombinationMethod(voting_method)
            except ValueError:
                self.logger.warning(f"Phương pháp bỏ phiếu '{voting_method}' không hợp lệ, sử dụng 'weighted_vote'")
                self.voting_method = AgentCombinationMethod.WEIGHTED_VOTE
        else:
            self.voting_method = voting_method
        
        self.performance_window = performance_window
        
        # Khởi tạo trọng số và theo dõi hiệu suất
        self.agent_weights = np.ones(len(self.agents)) / len(self.agents)  # Khởi tạo với trọng số bằng nhau
        self.agent_performances = [[] for _ in range(len(self.agents))]
        
        # Lưu trữ kinh nghiệm
        self.memory_buffer = []
        
        # Meta-model cho stacking (nếu cần)
        self.meta_model = None
        if self.ensemble_method == EnsembleMethod.STACKING:
            self._initialize_meta_model(meta_model_path)
        
        # Dictionary cho các hàm tùy chỉnh
        self.custom_functions = {
            "ensemble_fn": kwargs.get("custom_ensemble_fn", None),
            "selection_fn": kwargs.get("custom_selection_fn", None),
            "reward_fn": kwargs.get("custom_reward_fn", None)
        }
        
        # Tham số cho boosting
        self.agent_errors = np.zeros(len(self.agents))
        self.training_iteration = 0
        
        # Các biến theo dõi
        self.last_actions = []
        self.last_states = []
        self.last_rewards = []
        
        self.logger.info(
            f"Đã khởi tạo {self.__class__.__name__} với {len(self.agents)} agent cơ sở, "
            f"ensemble_method={self.ensemble_method.value}, "
            f"voting_method={self.voting_method.value}"
        )
    
    def act(self, state: np.ndarray, explore: bool = True) -> Union[int, np.ndarray]:
        """
        Chọn hành động dựa trên trạng thái hiện tại.
        
        Args:
            state: Trạng thái hiện tại
            explore: Có thực hiện khám phá không
            
        Returns:
            Hành động được chọn
        """
        # Lưu trạng thái hiện tại
        self.last_states.append(state)
        
        # Lấy hành động từ tất cả các agent
        agent_actions = []
        for agent in self.agents:
            try:
                action = agent.act(state, explore)
                agent_actions.append(action)
            except Exception as e:
                self.logger.warning(f"Lỗi khi lấy hành động từ agent '{agent.name}': {str(e)}")
                # Nếu gặp lỗi, sử dụng hành động trước đó nếu có
                if self.last_actions:
                    agent_actions.append(self.last_actions[-1])
                else:
                    # Nếu không có hành động trước đó, tạo một hành động ngẫu nhiên
                    if isinstance(self.action_dim, tuple):
                        # Continuous action space
                        action = np.random.uniform(-1, 1, self.action_dim)
                    else:
                        # Discrete action space
                        action = np.random.randint(0, self.action_dim)
                    agent_actions.append(action)
        
        # Tổng hợp hành động dựa trên phương pháp đã chọn
        if self.ensemble_method == EnsembleMethod.VOTING:
            action = self._voting_ensemble(agent_actions, explore)
        elif self.ensemble_method == EnsembleMethod.AVERAGING:
            action = self._averaging_ensemble(agent_actions, explore)
        elif self.ensemble_method == EnsembleMethod.BAGGING:
            action = self._bagging_ensemble(agent_actions, state, explore)
        elif self.ensemble_method == EnsembleMethod.STACKING:
            action = self._stacking_ensemble(agent_actions, state, explore)
        elif self.ensemble_method == EnsembleMethod.SWITCHING:
            action = self._switching_ensemble(agent_actions, state, explore)
        elif self.ensemble_method == EnsembleMethod.CASCADING:
            action = self._cascading_ensemble(agent_actions, state, explore)
        elif self.ensemble_method == EnsembleMethod.BOOSTING:
            action = self._boosting_ensemble(agent_actions, state, explore)
        elif self.ensemble_method == EnsembleMethod.CUSTOM and self.custom_functions["ensemble_fn"] is not None:
            try:
                action = self.custom_functions["ensemble_fn"](agent_actions, state, self.agent_weights, explore)
            except Exception as e:
                self.logger.error(f"Lỗi khi sử dụng hàm tổng hợp tùy chỉnh: {str(e)}")
                # Fallback to voting
                action = self._voting_ensemble(agent_actions, explore)
        else:
            # Mặc định là voting
            action = self._voting_ensemble(agent_actions, explore)
        
        # Lưu hành động đã chọn
        self.last_actions.append(action)
        self.step_count += 1
        
        return action
    
    def learn(self) -> Dict[str, float]:
        """
        Học từ bộ nhớ kinh nghiệm.
        
        Returns:
            Dict chứa thông tin về quá trình học (loss, v.v.)
        """
        # Đối với agent tổng hợp, việc học có thể bao gồm:
        # 1. Cập nhật meta-model (cho stacking)
        # 2. Cập nhật trọng số agent (cho boosting)
        # 3. Học cho các agent cơ sở (nếu cần)
        
        loss_info = {"loss": 0.0}
        
        # Cập nhật meta-model nếu có đủ dữ liệu
        if self.ensemble_method == EnsembleMethod.STACKING and self.meta_model is not None and len(self.memory_buffer) >= 64:
            loss_info = self._train_meta_model()
        
        # Cập nhật trọng số cho boosting nếu cần
        if self.ensemble_method == EnsembleMethod.BOOSTING and self.training_iteration > 0:
            self._update_boosting_weights()
        
        # Học cho các agent cơ sở (nếu được cấu hình)
        if self.kwargs.get("train_base_agents", False):
            for agent in self.agents:
                agent_loss_info = agent.learn()
                if agent_loss_info:
                    # Kết hợp thông tin loss từ các agent
                    for key, value in agent_loss_info.items():
                        if key not in loss_info:
                            loss_info[key] = value / len(self.agents)
                        else:
                            loss_info[key] += value / len(self.agents)
        
        return loss_info
    
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
        # Lưu trữ trải nghiệm
        experience = (state, action, reward, next_state, done)
        self.memory_buffer.append(experience)
        
        # Giới hạn kích thước bộ nhớ
        if len(self.memory_buffer) > 10000:
            self.memory_buffer.pop(0)
        
        # Lưu trải nghiệm cho các agent cơ sở (nếu được cấu hình)
        if self.kwargs.get("share_experience", True):
            for agent in self.agents:
                agent.remember(state, action, reward, next_state, done)
        
        # Lưu phần thưởng hiện tại
        self.last_rewards.append(reward)
        
        # Cập nhật hiệu suất của các agent
        if len(self.last_actions) >= 2:
            self._update_agent_performances(reward, done)
        
        # Cập nhật trọng số các agent
        self._update_agent_weights()
    
    def _initialize_meta_model(self, meta_model_path: Optional[Union[str, Path]] = None) -> None:
        """
        Khởi tạo meta-model cho phương pháp stacking.
        
        Args:
            meta_model_path: Đường dẫn đến meta-model (nếu có)
        """
        try:
            if meta_model_path and os.path.exists(meta_model_path):
                # Tải meta-model đã huấn luyện trước
                self.meta_model = tf.keras.models.load_model(meta_model_path)
                self.logger.info(f"Đã tải meta-model từ {meta_model_path}")
            else:
                # Tạo meta-model mới
                # Input sẽ là state + actions từ tất cả các agent
                if isinstance(self.state_dim, tuple):
                    state_flat_dim = np.prod(self.state_dim)
                else:
                    state_flat_dim = self.state_dim
                
                if isinstance(self.action_dim, tuple):
                    action_flat_dim = np.prod(self.action_dim)
                else:
                    action_flat_dim = 1  # Discrete action
                
                input_dim = state_flat_dim + (action_flat_dim * len(self.agents))
                
                # Tạo model
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(action_flat_dim)
                ])
                
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                    loss='mse'
                )
                
                self.meta_model = model
                self.logger.info("Đã khởi tạo meta-model mới cho stacking")
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo meta-model: {str(e)}")
            self.meta_model = None
    
    def _train_meta_model(self) -> Dict[str, float]:
        """
        Huấn luyện meta-model cho phương pháp stacking.
        
        Returns:
            Dict chứa thông tin về quá trình huấn luyện
        """
        if not self.meta_model or len(self.memory_buffer) < 64:
            return {"loss": 0.0}
        
        # Lấy batch từ memory buffer
        batch_size = min(64, len(self.memory_buffer))
        batch_indices = np.random.choice(len(self.memory_buffer), batch_size, replace=False)
        batch = [self.memory_buffer[idx] for idx in batch_indices]
        
        # Chuẩn bị dữ liệu huấn luyện
        X = []
        y = []
        
        for state, action, reward, next_state, done in batch:
            # Lấy hành động của tất cả agent cho state
            agent_actions = []
            for agent in self.agents:
                agent_action = agent.act(state, explore=False)
                if isinstance(agent_action, np.ndarray):
                    agent_actions.extend(agent_action.flatten())
                else:
                    agent_actions.append(agent_action)
            
            # Input: state + actions của tất cả agent
            if isinstance(state, np.ndarray):
                state_flat = state.flatten()
            else:
                state_flat = np.array([state])
            
            x = np.concatenate([state_flat, np.array(agent_actions)])
            
            # Output: action thực tế
            if isinstance(action, np.ndarray):
                y.append(action.flatten())
            else:
                y.append(np.array([action]))
        
        # Chuyển đổi sang numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Huấn luyện meta-model
        history = self.meta_model.fit(X, y, epochs=5, verbose=0)
        
        return {"loss": history.history['loss'][-1]}
    
    def _update_agent_performances(self, reward: float, done: bool) -> None:
        """
        Cập nhật hiệu suất của các agent dựa trên phần thưởng mới nhất.
        
        Args:
            reward: Phần thưởng nhận được
            done: True nếu episode kết thúc, False nếu không
        """
        # Cập nhật hiệu suất cho từng agent
        for i, agent in enumerate(self.agents):
            # Kiểm tra xem agent có dự đoán được hành động gần đây không
            if len(self.last_states) >= 2:
                prev_state = self.last_states[-2]
                agent_action = agent.act(prev_state, explore=False)
                actual_action = self.last_actions[-2]
                
                # Tính độ chính xác của dự đoán
                if isinstance(agent_action, np.ndarray) and isinstance(actual_action, np.ndarray):
                    # Đối với hành động liên tục, sử dụng độ tương tự (similarity)
                    # 1 - normalized distance
                    similarity = 1.0 - np.mean(np.abs(agent_action - actual_action)) / 2.0
                    agent_performance = similarity * reward
                else:
                    # Đối với hành động rời rạc
                    agent_performance = reward if agent_action == actual_action else 0.0
                
                # Lưu hiệu suất
                self.agent_performances[i].append(agent_performance)
                
                # Giới hạn kích thước cửa sổ hiệu suất
                if len(self.agent_performances[i]) > self.performance_window:
                    self.agent_performances[i].pop(0)
                
                # Cập nhật lỗi cho boosting
                if done:
                    # Tính lỗi trung bình trong episode
                    if isinstance(agent_action, np.ndarray) and isinstance(actual_action, np.ndarray):
                        self.agent_errors[i] = np.mean(np.abs(agent_action - actual_action))
                    else:
                        self.agent_errors[i] = 0.0 if agent_action == actual_action else 1.0
                    
                    self.training_iteration += 1
    
    def _update_agent_weights(self) -> None:
        """
        Cập nhật trọng số cho các agent dựa trên hiệu suất.
        """
        # Cập nhật trọng số cho từng agent
        for i, performances in enumerate(self.agent_performances):
            if performances:
                # Tính trung bình hiệu suất trong cửa sổ
                avg_performance = np.mean(performances)
                
                # Điều chỉnh trọng số
                if avg_performance < 0:
                    # Nếu hiệu suất âm, cho trọng số nhỏ
                    self.agent_weights[i] = 0.1
                else:
                    # Nếu hiệu suất dương, tỉ lệ thuận với hiệu suất
                    self.agent_weights[i] = 1.0 + avg_performance
        
        # Chuẩn hóa trọng số
        total_weight = np.sum(self.agent_weights)
        if total_weight > 0:
            self.agent_weights = self.agent_weights / total_weight
    
    def _update_boosting_weights(self) -> None:
        """
        Cập nhật trọng số cho phương pháp boosting.
        """
        # Cập nhật trọng số theo thuật toán AdaBoost
        for i in range(len(self.agents)):
            # Tính hệ số alpha dựa trên lỗi
            error = self.agent_errors[i]
            if error >= 1.0 or error <= 0.0:
                alpha = 1.0
            else:
                alpha = 0.5 * np.log((1.0 - error) / max(error, 1e-10))
            
            # Cập nhật trọng số
            self.agent_weights[i] = np.exp(-alpha * error)
        
        # Chuẩn hóa trọng số
        total_weight = np.sum(self.agent_weights)
        if total_weight > 0:
            self.agent_weights = self.agent_weights / total_weight
    
    def _voting_ensemble(self, agent_actions: List[Union[int, np.ndarray]], explore: bool = True) -> Union[int, np.ndarray]:
        """
        Tổng hợp hành động bằng phương pháp bỏ phiếu.
        
        Args:
            agent_actions: Danh sách hành động từ các agent
            explore: Có thực hiện khám phá không
            
        Returns:
            Hành động được kết hợp
        """
        # Kiểm tra kiểu hành động
        if isinstance(agent_actions[0], np.ndarray):
            # Xử lý hành động dạng mảng
            if self.voting_method == AgentCombinationMethod.WEIGHTED_VOTE:
                # Lấy trung bình có trọng số
                action = np.zeros_like(agent_actions[0], dtype=float)
                for i, a in enumerate(agent_actions):
                    action += a * self.agent_weights[i]
            else:
                # Lấy trung bình thông thường
                action = np.mean(agent_actions, axis=0)
            
            # Thêm nhiễu khám phá nếu cần
            if explore:
                action = action + np.random.normal(0, 0.1, size=action.shape)
        else:
            # Xử lý hành động rời rạc
            if self.voting_method == AgentCombinationMethod.MAJORITY_VOTE:
                # Bỏ phiếu đa số
                action_counts = {}
                for action in agent_actions:
                    action_counts[action] = action_counts.get(action, 0) + 1
                
                # Tìm hành động được bỏ phiếu nhiều nhất
                action = max(action_counts, key=action_counts.get)
            elif self.voting_method == AgentCombinationMethod.WEIGHTED_VOTE:
                # Bỏ phiếu có trọng số
                action_weights = {}
                for i, action in enumerate(agent_actions):
                    action_weights[action] = action_weights.get(action, 0) + self.agent_weights[i]
                
                # Tìm hành động có tổng trọng số cao nhất
                action = max(action_weights, key=action_weights.get)
            else:
                # Mặc định là đa số
                from collections import Counter
                action = Counter(agent_actions).most_common(1)[0][0]
            
            # Thêm khám phá ngẫu nhiên
            if explore and random.random() < 0.1:
                action = random.randint(0, self.action_dim - 1)
        
        return action
    
    def _averaging_ensemble(self, agent_actions: List[Union[int, np.ndarray]], explore: bool = True) -> Union[int, np.ndarray]:
        """
        Tổng hợp hành động bằng phương pháp trung bình hóa.
        
        Args:
            agent_actions: Danh sách hành động từ các agent
            explore: Có thực hiện khám phá không
            
        Returns:
            Hành động được kết hợp
        """
        # Kiểm tra kiểu hành động
        if isinstance(agent_actions[0], np.ndarray):
            # Xử lý hành động dạng mảng
            # Lấy trung bình có trọng số
            action = np.zeros_like(agent_actions[0], dtype=float)
            for i, a in enumerate(agent_actions):
                action += a * self.agent_weights[i]
            
            # Thêm nhiễu khám phá
            if explore:
                noise_scale = 0.1 * (1.0 / (1.0 + 0.1 * self.step_count))  # Giảm dần
                action = action + np.random.normal(0, noise_scale, size=action.shape)
        else:
            # Xử lý hành động rời rạc
            # Đối với hành động rời rạc, lấy trung bình và làm tròn
            weighted_sum = 0.0
            for i, a in enumerate(agent_actions):
                weighted_sum += a * self.agent_weights[i]
            
            action = int(round(weighted_sum))
            
            # Đảm bảo action nằm trong phạm vi hợp lệ
            action = max(0, min(action, self.action_dim - 1))
            
            # Khám phá
            if explore and random.random() < 0.1:
                action = random.randint(0, self.action_dim - 1)
        
        return action
    
    def _bagging_ensemble(self, agent_actions: List[Union[int, np.ndarray]], state: np.ndarray, explore: bool = True) -> Union[int, np.ndarray]:
        """
        Tổng hợp hành động bằng phương pháp bagging.
        
        Args:
            agent_actions: Danh sách hành động từ các agent
            state: Trạng thái hiện tại
            explore: Có thực hiện khám phá không
            
        Returns:
            Hành động được kết hợp
        """
        # Bagging bao gồm:
        # 1. Chọn ngẫu nhiên một tập hợp con các agent (với thay thế)
        # 2. Lấy quyết định từ tập hợp con đó
        # 3. Kết hợp các quyết định
        
        # Số lượng agent trong tập hợp con
        n_agents = len(self.agents)
        bag_size = max(1, n_agents // 2)
        
        # Chọn ngẫu nhiên các agent (với thay thế)
        bag_indices = np.random.choice(n_agents, bag_size, replace=True)
        
        # Lấy hành động từ các agent được chọn
        bag_actions = [agent_actions[i] for i in bag_indices]
        bag_weights = [self.agent_weights[i] for i in bag_indices]
        
        # Chuẩn hóa trọng số
        total_weight = sum(bag_weights)
        if total_weight > 0:
            bag_weights = [w / total_weight for w in bag_weights]
        
        # Kết hợp hành động
        if isinstance(bag_actions[0], np.ndarray):
            # Xử lý hành động dạng mảng
            action = np.zeros_like(bag_actions[0], dtype=float)
            for i, a in enumerate(bag_actions):
                action += a * bag_weights[i]
            
            # Thêm nhiễu khám phá
            if explore:
                action = action + np.random.normal(0, 0.1, size=action.shape)
        else:
            # Xử lý hành động rời rạc
            action_weights = {}
            for i, a in enumerate(bag_actions):
                action_weights[a] = action_weights.get(a, 0) + bag_weights[i]
            
            # Tìm hành động có tổng trọng số cao nhất
            action = max(action_weights, key=action_weights.get)
            
            # Khám phá
            if explore and random.random() < 0.1:
                action = random.randint(0, self.action_dim - 1)
        
        return action
    
    def _stacking_ensemble(self, agent_actions: List[Union[int, np.ndarray]], state: np.ndarray, explore: bool = True) -> Union[int, np.ndarray]:
        """
        Tổng hợp hành động bằng phương pháp stacking.
        
        Args:
            agent_actions: Danh sách hành động từ các agent
            state: Trạng thái hiện tại
            explore: Có thực hiện khám phá không
            
        Returns:
            Hành động được kết hợp
        """
        # Nếu meta-model chưa được khởi tạo hoặc chưa đủ dữ liệu, sử dụng phương pháp voting
        if self.meta_model is None or len(self.memory_buffer) < 100:
            return self._voting_ensemble(agent_actions, explore)
        
        try:
            # Chuẩn bị input cho meta-model
            if isinstance(state, np.ndarray):
                state_flat = state.flatten()
            else:
                state_flat = np.array([state])
            
            # Chuyển đổi hành động của các agent thành vector
            agent_actions_flat = []
            for action in agent_actions:
                if isinstance(action, np.ndarray):
                    agent_actions_flat.extend(action.flatten())
                else:
                    agent_actions_flat.append(action)
            
            # Kết hợp state và hành động
            x = np.concatenate([state_flat, np.array(agent_actions_flat)])
            x = np.expand_dims(x, axis=0)  # Thêm chiều batch
            
            # Dự đoán hành động bằng meta-model
            predicted_action = self.meta_model.predict(x, verbose=0)[0]
            
            # Xử lý kết quả dự đoán
            if isinstance(agent_actions[0], np.ndarray):
                # Reshape về kích thước hành động ban đầu
                action = predicted_action.reshape(agent_actions[0].shape)
                
                # Thêm nhiễu khám phá
                if explore:
                    action = action + np.random.normal(0, 0.1, size=action.shape)
            else:
                # Đối với hành động rời rạc, làm tròn
                action = int(round(predicted_action[0]))
                
                # Đảm bảo action nằm trong phạm vi hợp lệ
                action = max(0, min(action, self.action_dim - 1))
                
                # Khám phá
                if explore and random.random() < 0.1:
                    action = random.randint(0, self.action_dim - 1)
            
            return action
        except Exception as e:
            self.logger.error(f"Lỗi khi sử dụng meta-model: {str(e)}")
            # Fallback to voting
            return self._voting_ensemble(agent_actions, explore)
    
    def _switching_ensemble(self, agent_actions: List[Union[int, np.ndarray]], state: np.ndarray, explore: bool = True) -> Union[int, np.ndarray]:
        """
        Tổng hợp hành động bằng phương pháp switching.
        
        Args:
            agent_actions: Danh sách hành động từ các agent
            state: Trạng thái hiện tại
            explore: Có thực hiện khám phá không
            
        Returns:
            Hành động được kết hợp
        """
        # Switching đổi giữa các agent dựa trên hiệu suất và ngữ cảnh
        
        # Tính điểm phù hợp cho từng agent dựa trên hiệu suất gần đây
        agent_scores = np.zeros(len(self.agents))
        
        for i, performances in enumerate(self.agent_performances):
            if performances:
                # Tính trung bình hiệu suất gần đây (nghiêng về hiệu suất gần nhất)
                weights = np.linspace(0.5, 1.0, len(performances))
                weights = weights / np.sum(weights)
                avg_performance = np.sum(np.array(performances) * weights)
                
                agent_scores[i] = avg_performance
        
        # Phân tích đặc điểm của state để điều chỉnh điểm
        # Ví dụ: một số agent có thể phù hợp hơn với trạng thái nhất định
        
        # Chọn agent có điểm cao nhất
        if np.all(agent_scores <= 0):
            # Nếu tất cả agent có điểm không tốt, sử dụng phương pháp voting
            return self._voting_ensemble(agent_actions, explore)
        
        best_agent_idx = np.argmax(agent_scores)
        action = agent_actions[best_agent_idx]
        
        # Thêm khám phá nếu cần
        if explore:
            if isinstance(action, np.ndarray):
                action = action + np.random.normal(0, 0.1, size=action.shape)
            else:
                if random.random() < 0.1:
                    action = random.randint(0, self.action_dim - 1)
        
        return action
    
    def _cascading_ensemble(self, agent_actions: List[Union[int, np.ndarray]], state: np.ndarray, explore: bool = True) -> Union[int, np.ndarray]:
        """
        Tổng hợp hành động bằng phương pháp cascading.
        
        Args:
            agent_actions: Danh sách hành động từ các agent
            state: Trạng thái hiện tại
            explore: Có thực hiện khám phá không
            
        Returns:
            Hành động được kết hợp
        """
        # Cascading sử dụng các agent theo thứ tự, cho đến khi tìm được hành động với độ tự tin cao
        
        # Sắp xếp agent theo hiệu suất
        agent_indices = list(range(len(self.agents)))
        # Sắp xếp theo trọng số giảm dần
        agent_indices = sorted(agent_indices, key=lambda i: self.agent_weights[i], reverse=True)
        
        # Ngưỡng độ tự tin (có thể điều chỉnh theo thời gian)
        confidence_threshold = 0.7
        
        # Kiểm tra từng agent theo thứ tự
        for idx in agent_indices:
            agent = self.agents[idx]
            action = agent_actions[idx]
            
            # Nếu agent hỗ trợ act_with_confidence, sử dụng nó
            if hasattr(agent, 'act_with_confidence'):
                try:
                    _, confidence = agent.act_with_confidence(state, explore=False)
                    
                    if confidence >= confidence_threshold:
                        # Nếu độ tự tin đủ cao, sử dụng hành động này
                        if explore and random.random() < 0.1:
                            # Thêm khám phá
                            if isinstance(action, np.ndarray):
                                action = action + np.random.normal(0, 0.1, size=action.shape)
                            else:
                                if random.random() < 0.1:
                                    action = random.randint(0, self.action_dim - 1)
                        
                        return action
                except Exception:
                    # Nếu gặp lỗi, tiếp tục với agent tiếp theo
                    continue
        
        # Nếu không có agent nào đủ tự tin, sử dụng phương pháp voting
        return self._voting_ensemble(agent_actions, explore)
    
    def _boosting_ensemble(self, agent_actions: List[Union[int, np.ndarray]], state: np.ndarray, explore: bool = True) -> Union[int, np.ndarray]:
        """
        Tổng hợp hành động bằng phương pháp boosting.
        
        Args:
            agent_actions: Danh sách hành động từ các agent
            state: Trạng thái hiện tại
            explore: Có thực hiện khám phá không
            
        Returns:
            Hành động được kết hợp
        """
        # Boosting kết hợp các agent theo trọng số đã được điều chỉnh
        # trong quá trình huấn luyện
        
        # Kiểm tra xem đã có đủ dữ liệu chưa
        if self.training_iteration < 1:
            # Nếu chưa đủ dữ liệu để điều chỉnh trọng số, sử dụng phương pháp voting
            return self._voting_ensemble(agent_actions, explore)
        
        # Kết hợp hành động theo trọng số boosting
        if isinstance(agent_actions[0], np.ndarray):
            # Xử lý hành động dạng mảng
            action = np.zeros_like(agent_actions[0], dtype=float)
            for i, a in enumerate(agent_actions):
                action += a * self.agent_weights[i]
            
            # Thêm nhiễu khám phá
            if explore:
                action = action + np.random.normal(0, 0.1, size=action.shape)
        else:
            # Xử lý hành động rời rạc
            action_weights = {}
            for i, a in enumerate(agent_actions):
                action_weights[a] = action_weights.get(a, 0) + self.agent_weights[i]
            
            # Tìm hành động có tổng trọng số cao nhất
            action = max(action_weights, key=action_weights.get)
            
            # Khám phá
            if explore and random.random() < 0.1:
                action = random.randint(0, self.action_dim - 1)
        
        return action
    
    def act_with_confidence(self, state: np.ndarray, explore: bool = True) -> Tuple[Union[int, np.ndarray], float]:
        """
        Chọn hành động kèm theo độ tự tin.
        
        Args:
            state: Trạng thái hiện tại
            explore: Có thực hiện khám phá không
            
        Returns:
            Tuple (hành động, độ tự tin)
        """
        # Lấy hành động và độ tự tin từ tất cả các agent
        agent_actions = []
        agent_confidences = []
        
        for agent in self.agents:
            try:
                if hasattr(agent, 'act_with_confidence'):
                    action, confidence = agent.act_with_confidence(state, explore)
                else:
                    action = agent.act(state, explore)
                    # Giả định độ tự tin là 1.0 nếu không có
                    confidence = 1.0
                
                agent_actions.append(action)
                agent_confidences.append(confidence)
            except Exception as e:
                self.logger.warning(f"Lỗi khi lấy hành động từ agent '{agent.name}': {str(e)}")
                # Bỏ qua agent lỗi
        
        if not agent_actions:
            # Nếu không có agent nào thành công, trả về hành động ngẫu nhiên
            if isinstance(self.action_dim, tuple):
                # Continuous action space
                action = np.random.uniform(-1, 1, self.action_dim)
            else:
                # Discrete action space
                action = np.random.randint(0, self.action_dim)
            
            return action, 0.1  # Độ tự tin thấp
        
        # Tính độ tự tin dựa trên độ đồng thuận của các agent
        avg_confidence = np.mean(agent_confidences)
        
        # Độ đồng thuận: kiểm tra các agent có đưa ra hành động tương tự không
        if isinstance(agent_actions[0], np.ndarray):
            # Tính độ lệch chuẩn của các hành động
            action_std = np.std([a.flatten() for a in agent_actions], axis=0)
            # Độ đồng thuận ngược với độ lệch chuẩn
            consensus = np.exp(-np.mean(action_std))
        else:
            # Đếm số lượng mỗi hành động
            from collections import Counter
            action_counts = Counter(agent_actions)
            most_common_count = action_counts.most_common(1)[0][1]
            # Độ đồng thuận: tỉ lệ agent chọn hành động phổ biến nhất
            consensus = most_common_count / len(agent_actions)
        
        # Kết hợp độ tự tin trung bình và độ đồng thuận
        combined_confidence = 0.5 * avg_confidence + 0.5 * consensus
        
        # Lấy hành động theo phương pháp ensemble đã chọn
        action = self.act(state, explore)
        
        return action, combined_confidence
    
    def _save_model_impl(self, path: Union[str, Path]) -> None:
        """
        Triển khai chi tiết việc lưu mô hình.
        
        Args:
            path: Đường dẫn lưu mô hình
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Lưu meta-model nếu có
        if self.meta_model is not None:
            meta_model_path = path.parent / f"{path.name}_meta_model"
            self.meta_model.save(meta_model_path)
        
        # Lưu trạng thái và cấu hình
        state = {
            "agent_weights": self.agent_weights.tolist(),
            "ensemble_method": self.ensemble_method.value,
            "voting_method": self.voting_method.value,
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "training_iteration": self.training_iteration,
            "agent_errors": self.agent_errors.tolist(),
            "created_at": time.time()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4, ensure_ascii=False)
        
        # Lưu từng agent con nếu được cấu hình
        if self.kwargs.get("save_base_agents", True):
            for i, agent in enumerate(self.agents):
                agent_path = path.parent / f"{path.name}_agent_{i}"
                try:
                    agent._save_model_impl(agent_path)
                except Exception as e:
                    self.logger.warning(f"Không thể lưu agent {i}: {str(e)}")
        
        self.logger.info(f"Đã lưu EnsembleAgent tại {path}")
    
    def _load_model_impl(self, path: Union[str, Path]) -> bool:
        """
        Triển khai chi tiết việc tải mô hình.
        
        Args:
            path: Đường dẫn tải mô hình
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        path = Path(path)
        if not path.exists():
            self.logger.warning(f"Không tìm thấy file mô hình tại {path}")
            return False
        
        try:
            # Tải trạng thái và cấu hình
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            if "agent_weights" in state:
                self.agent_weights = np.array(state["agent_weights"])
            
            if "ensemble_method" in state:
                try:
                    self.ensemble_method = EnsembleMethod(state["ensemble_method"])
                except ValueError:
                    self.logger.warning(f"Phương pháp tổng hợp '{state['ensemble_method']}' không hợp lệ, giữ nguyên giá trị hiện tại")
            
            if "voting_method" in state:
                try:
                    self.voting_method = AgentCombinationMethod(state["voting_method"])
                except ValueError:
                    self.logger.warning(f"Phương pháp bỏ phiếu '{state['voting_method']}' không hợp lệ, giữ nguyên giá trị hiện tại")
            
            if "step_count" in state:
                self.step_count = state["step_count"]
            
            if "episode_count" in state:
                self.episode_count = state["episode_count"]
            
            if "training_iteration" in state:
                self.training_iteration = state["training_iteration"]
            
            if "agent_errors" in state:
                self.agent_errors = np.array(state["agent_errors"])
            
            # Tải meta-model nếu có
            meta_model_path = path.parent / f"{path.name}_meta_model"
            if meta_model_path.exists() and self.ensemble_method == EnsembleMethod.STACKING:
                try:
                    self.meta_model = tf.keras.models.load_model(meta_model_path)
                    self.logger.info(f"Đã tải meta-model từ {meta_model_path}")
                except Exception as e:
                    self.logger.warning(f"Không thể tải meta-model: {str(e)}")
            
            # Tải từng agent con nếu được cấu hình
            if self.kwargs.get("load_base_agents", True):
                for i, agent in enumerate(self.agents):
                    agent_path = path.parent / f"{path.name}_agent_{i}"
                    if agent_path.exists():
                        try:
                            agent._load_model_impl(agent_path)
                        except Exception as e:
                            self.logger.warning(f"Không thể tải agent {i}: {str(e)}")
            
            self.logger.info(f"Đã tải EnsembleAgent từ {path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tải mô hình: {str(e)}")
            return False
    
    def get_agent(self, idx: int) -> Optional[BaseAgent]:
        """
        Lấy agent theo chỉ số.
        
        Args:
            idx: Chỉ số của agent
            
        Returns:
            Agent tại chỉ số đã cho, None nếu không tìm thấy
        """
        if idx < 0 or idx >= len(self.agents):
            self.logger.warning(f"Chỉ số agent {idx} không hợp lệ, phạm vi hợp lệ: 0-{len(self.agents)-1}")
            return None
        
        return self.agents[idx]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê hiệu suất của ensemble và các agent cơ sở.
        
        Returns:
            Dict chứa thống kê hiệu suất
        """
        stats = {
            "ensemble": {
                "ensemble_method": self.ensemble_method.value,
                "voting_method": self.voting_method.value,
                "step_count": self.step_count,
                "episode_count": self.episode_count,
                "training_iteration": self.training_iteration
            },
            "agents": {}
        }
        
        # Thống kê chi tiết cho từng agent
        for i, agent in enumerate(self.agents):
            agent_name = agent.name
            performances = self.agent_performances[i]
            weight = self.agent_weights[i]
            
            if performances:
                avg_performance = np.mean(performances)
                min_performance = np.min(performances)
                max_performance = np.max(performances)
                std_performance = np.std(performances)
            else:
                avg_performance = 0.0
                min_performance = 0.0
                max_performance = 0.0
                std_performance = 0.0
            
            stats["agents"][agent_name] = {
                "index": i,
                "weight": float(weight),
                "avg_performance": float(avg_performance),
                "min_performance": float(min_performance),
                "max_performance": float(max_performance),
                "std_performance": float(std_performance),
                "error": float(self.agent_errors[i])
            }
        
        return stats
    
    def reset(self) -> None:
        """
        Đặt lại trạng thái của agent.
        """
        # Đặt lại các biến theo dõi
        super().reset()
        
        # Đặt lại trạng thái của các agent cơ sở
        for agent in self.agents:
            agent.reset()
        
        # Làm trống các danh sách lưu trữ
        self.last_actions = []
        self.last_states = []
        self.last_rewards = []
        
        # Đặt lại trọng số
        self.agent_weights = np.ones(len(self.agents)) / len(self.agents)
        
        # Đặt lại biến theo dõi khác
        self.training_iteration = 0
        self.agent_errors = np.zeros(len(self.agents))
        
        self.logger.info(f"Đã đặt lại trạng thái của {self.__class__.__name__}")