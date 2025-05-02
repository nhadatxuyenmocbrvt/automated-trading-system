"""
Lớp agent cơ sở.
File này định nghĩa lớp BaseAgent làm nền tảng cho tất cả các agent,
cung cấp các phương thức và thuộc tính cơ bản mà tất cả các agent cần có.
"""

import os
import time
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
from pathlib import Path

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from environments.base_environment import BaseEnvironment

class BaseAgent(ABC):
    """
    Lớp cơ sở cho tất cả các agent.
    Định nghĩa giao diện chung mà tất cả các agent phải tuân theo.
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
        save_dir: Optional[Union[str, Path]] = None,
        name: str = "base_agent",
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Khởi tạo agent cơ sở.
        
        Args:
            state_dim: Kích thước không gian trạng thái (số chiều)
            action_dim: Kích thước không gian hành động
            env: Môi trường tương tác
            learning_rate: Tốc độ học
            gamma: Hệ số giảm phần thưởng
            epsilon: Tham số khám phá ban đầu (cho ε-greedy)
            epsilon_min: Giá trị epsilon tối thiểu
            epsilon_decay: Tốc độ giảm epsilon sau mỗi bước
            batch_size: Kích thước batch huấn luyện
            memory_size: Kích thước bộ nhớ kinh nghiệm
            update_target_freq: Tần suất cập nhật mạng mục tiêu (nếu có)
            save_dir: Thư mục lưu mô hình
            name: Tên của agent
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("agent")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập các thuộc tính
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.update_target_freq = update_target_freq
        self.name = name
        
        # Khởi tạo biến đếm bước
        self.step_count = 0
        self.train_count = 0
        self.episode_count = 0
        
        # Lưu thông tin cấu hình
        self.config = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "epsilon": epsilon,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "batch_size": batch_size,
            "memory_size": memory_size,
            "update_target_freq": update_target_freq,
            "name": name,
            **kwargs
        }
        
        # Thư mục lưu mô hình
        if save_dir is None:
            save_dir = self.system_config.get("agent.checkpoint_dir")
        self.save_dir = Path(save_dir) / self.name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Lịch sử huấn luyện
        self.training_history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": [],
            "epsilons": []
        }
        
        self.logger.info(f"Đã khởi tạo {self.__class__.__name__} với state_dim={state_dim}, action_dim={action_dim}")
    
    @abstractmethod
    def act(self, state: np.ndarray, explore: bool = True) -> Union[int, np.ndarray]:
        """
        Chọn hành động dựa trên trạng thái hiện tại.
        
        Args:
            state: Trạng thái hiện tại
            explore: True nếu agent nên khám phá, False nếu chỉ khai thác
            
        Returns:
            Hành động được chọn
        """
        pass
    
    @abstractmethod
    def learn(self) -> Dict[str, float]:
        """
        Học từ bộ nhớ kinh nghiệm.
        
        Returns:
            Dict chứa thông tin về quá trình học (loss, v.v.)
        """
        pass
    
    @abstractmethod
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
        pass
    
    def update_epsilon(self) -> None:
        """
        Cập nhật giá trị epsilon cho chiến lược khám phá.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Lưu mô hình của agent.
        
        Args:
            path: Đường dẫn lưu mô hình. Nếu None, sử dụng đường dẫn mặc định
        """
        if path is None:
            path = self.save_dir / f"{self.name}_model.h5"
        
        self.logger.info(f"Lưu mô hình tại {path}")
        
        # Lớp con sẽ thực hiện chi tiết việc lưu mô hình
        self._save_model_impl(path)
        
        # Lưu cấu hình và lịch sử huấn luyện
        config_path = self.save_dir / f"{self.name}_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        
        history_path = self.save_dir / f"{self.name}_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            # Chuyển đổi ndarray thành list
            history = {
                key: [float(val) if isinstance(val, np.number) else val 
                      for val in values]
                for key, values in self.training_history.items()
            }
            json.dump(history, f, indent=4, ensure_ascii=False)
    
    def load_model(self, path: Optional[Union[str, Path]] = None) -> bool:
        """
        Tải mô hình của agent.
        
        Args:
            path: Đường dẫn tải mô hình. Nếu None, sử dụng đường dẫn mặc định
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        if path is None:
            path = self.save_dir / f"{self.name}_model.h5"
        
        path = Path(path)
        if not path.exists():
            self.logger.warning(f"Không tìm thấy mô hình tại {path}")
            return False
        
        self.logger.info(f"Tải mô hình từ {path}")
        
        # Tải cấu hình
        config_path = path.parent / f"{self.name}_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Cập nhật cấu hình hiện tại với giá trị từ file
                for key, value in config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
        
        # Tải lịch sử huấn luyện
        history_path = path.parent / f"{self.name}_history.json"
        if history_path.exists():
            with open(history_path, 'r', encoding='utf-8') as f:
                self.training_history = json.load(f)
        
        # Lớp con sẽ thực hiện chi tiết việc tải mô hình
        return self._load_model_impl(path)
    
    def reset(self) -> None:
        """
        Đặt lại trạng thái của agent.
        """
        self.step_count = 0
        self.epsilon = self.config["epsilon"]  # Đặt lại epsilon ban đầu
    
    def train(self, env: BaseEnvironment, num_episodes: int, max_steps: int = 1000,
             save_freq: int = 10, log_freq: int = 1, render: bool = False,
             eval_freq: int = 10, eval_episodes: int = 5) -> Dict[str, List[float]]:
        """
        Huấn luyện agent trên môi trường.
        
        Args:
            env: Môi trường huấn luyện
            num_episodes: Số lượng episode huấn luyện
            max_steps: Số bước tối đa trong mỗi episode
            save_freq: Tần suất lưu mô hình (mỗi bao nhiêu episode)
            log_freq: Tần suất ghi log (mỗi bao nhiêu episode)
            render: True để hiển thị môi trường trong quá trình huấn luyện
            eval_freq: Tần suất đánh giá (mỗi bao nhiêu episode)
            eval_episodes: Số lượng episode đánh giá
            
        Returns:
            Dict chứa lịch sử huấn luyện
        """
        self.logger.info(f"Bắt đầu huấn luyện {self.name} trên {env.__class__.__name__} với {num_episodes} episodes")
        
        for episode in range(1, num_episodes + 1):
            state = env.reset()
            episode_reward = 0
            episode_loss = []
            
            for step in range(max_steps):
                # Chọn hành động
                action = self.act(state)
                
                # Thực hiện hành động
                next_state, reward, done, info = env.step(action)
                
                # Lưu trữ kinh nghiệm
                self.remember(state, action, reward, next_state, done)
                
                # Học từ kinh nghiệm
                loss_info = self.learn()
                if loss_info and "loss" in loss_info:
                    episode_loss.append(loss_info["loss"])
                
                # Cập nhật trạng thái
                state = next_state
                episode_reward += reward
                self.step_count += 1
                
                # Hiển thị môi trường nếu cần
                if render:
                    env.render()
                
                # Kiểm tra kết thúc episode
                if done:
                    break
            
            # Cập nhật epsilon
            self.update_epsilon()
            
            # Lưu lịch sử huấn luyện
            self.training_history["episode_rewards"].append(episode_reward)
            self.training_history["episode_lengths"].append(step + 1)
            self.training_history["epsilons"].append(self.epsilon)
            
            if episode_loss:
                avg_loss = sum(episode_loss) / len(episode_loss)
                self.training_history["losses"].append(avg_loss)
            else:
                self.training_history["losses"].append(0.0)
            
            # Ghi log
            if episode % log_freq == 0:
                self.logger.info(
                    f"Episode {episode}/{num_episodes}, "
                    f"Reward: {episode_reward:.2f}, "
                    f"Length: {step+1}, "
                    f"Epsilon: {self.epsilon:.4f}, "
                    f"Loss: {self.training_history['losses'][-1]:.4f}"
                )
            
            # Lưu mô hình
            if episode % save_freq == 0:
                self.save_model()
            
            # Đánh giá agent
            if episode % eval_freq == 0:
                eval_rewards = self.evaluate(env, eval_episodes)
                self.logger.info(
                    f"Đánh giá sau episode {episode}: "
                    f"Avg Reward: {np.mean(eval_rewards):.2f}, "
                    f"Min: {np.min(eval_rewards):.2f}, "
                    f"Max: {np.max(eval_rewards):.2f}"
                )
            
            self.episode_count += 1
        
        # Lưu mô hình cuối cùng
        self.save_model()
        
        self.logger.info(f"Hoàn thành huấn luyện {self.name} sau {num_episodes} episodes")
        return self.training_history
    
    def evaluate(self, env: BaseEnvironment, num_episodes: int = 5, 
                max_steps: int = 1000, render: bool = False) -> List[float]:
        """
        Đánh giá agent trên môi trường.
        
        Args:
            env: Môi trường đánh giá
            num_episodes: Số lượng episode đánh giá
            max_steps: Số bước tối đa trong mỗi episode
            render: True để hiển thị môi trường trong quá trình đánh giá
            
        Returns:
            Danh sách phần thưởng của các episode
        """
        self.logger.info(f"Đánh giá {self.name} trên {env.__class__.__name__} với {num_episodes} episodes")
        
        rewards = []
        
        for episode in range(1, num_episodes + 1):
            state = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # Chọn hành động không khám phá
                action = self.act(state, explore=False)
                
                # Thực hiện hành động
                next_state, reward, done, info = env.step(action)
                
                # Cập nhật trạng thái
                state = next_state
                episode_reward += reward
                
                # Hiển thị môi trường nếu cần
                if render:
                    env.render()
                
                # Kiểm tra kết thúc episode
                if done:
                    break
            
            rewards.append(episode_reward)
            
            self.logger.debug(f"Episode đánh giá {episode}/{num_episodes}, Reward: {episode_reward:.2f}")
        
        return rewards
    
    @abstractmethod
    def _save_model_impl(self, path: Union[str, Path]) -> None:
        """
        Triển khai cụ thể của việc lưu mô hình. Được ghi đè bởi lớp con.
        
        Args:
            path: Đường dẫn lưu mô hình
        """
        pass
    
    @abstractmethod
    def _load_model_impl(self, path: Union[str, Path]) -> bool:
        """
        Triển khai cụ thể của việc tải mô hình. Được ghi đè bởi lớp con.
        
        Args:
            path: Đường dẫn tải mô hình
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        pass