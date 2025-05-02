"""
Pipeline huấn luyện agent.
File này định nghĩa lớp Trainer để quản lý quá trình huấn luyện agents,
bao gồm quản lý dữ liệu, tối ưu hóa siêu tham số, và theo dõi quá trình huấn luyện.
"""

import os
import time
import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime
import shutil
import concurrent.futures
from functools import partial
import matplotlib.pyplot as plt

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import get_logger
from config.system_config import get_system_config, MODEL_DIR
from environments.base_environment import BaseEnvironment
from models.agents.base_agent import BaseAgent
from models.training_pipeline.experience_buffer import ExperienceBuffer

class Trainer:
    """
    Lớp quản lý việc huấn luyện agents.
    Cung cấp các phương thức cho việc thiết lập, huấn luyện, đánh giá 
    và lưu trữ các mô hình.
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        env: BaseEnvironment,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None,
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
        save_best_only: bool = True,
        early_stopping: bool = True,
        patience: int = 10,
        verbose: int = 1,
        **kwargs
    ):
        """
        Khởi tạo trainer.
        
        Args:
            agent: Agent cần huấn luyện
            env: Môi trường huấn luyện
            config: Cấu hình huấn luyện
            output_dir: Thư mục đầu ra cho kết quả huấn luyện
            logger: Logger tùy chỉnh
            experiment_name: Tên thí nghiệm
            use_tensorboard: Sử dụng TensorBoard để theo dõi
            save_best_only: Chỉ lưu mô hình tốt nhất
            early_stopping: Sử dụng early stopping
            patience: Số epochs chờ trước khi dừng sớm
            verbose: Mức độ chi tiết log (0: không log, 1: log cơ bản, 2: log chi tiết)
        """
        # Thiết lập logger
        self.logger = logger or get_logger("trainer")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập các thành phần chính
        self.agent = agent
        self.env = env
        
        # Thiết lập các tham số
        self.experiment_name = experiment_name or f"{agent.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.use_tensorboard = use_tensorboard
        self.save_best_only = save_best_only
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        
        # Thiết lập thư mục đầu ra
        if output_dir is None:
            self.output_dir = MODEL_DIR / 'training_results' / self.experiment_name
        else:
            self.output_dir = Path(output_dir) / self.experiment_name
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập cấu hình mặc định nếu không được cung cấp
        if config is None:
            config = self._get_default_config()
        self.config = config
        
        # Lưu các tham số bổ sung
        self.kwargs = kwargs
        
        # Khởi tạo biến theo dõi
        self.current_episode = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        self.best_episode = 0
        self.no_improvement_count = 0
        self.training_complete = False
        
        # Lịch sử huấn luyện
        self.history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": [],
            "eval_rewards": [],
            "timestamps": []
        }
        
        # Thiết lập TensorBoard nếu cần
        if self.use_tensorboard:
            tensorboard_dir = self.output_dir / 'tensorboard'
            tensorboard_dir.mkdir(exist_ok=True)
            self.tensorboard_writer = tf.summary.create_file_writer(str(tensorboard_dir))
        else:
            self.tensorboard_writer = None
        
        # Thiết lập Experience Buffer nếu cần
        if 'use_experience_buffer' in kwargs and kwargs['use_experience_buffer']:
            buffer_size = kwargs.get('buffer_size', 10000)
            self.experience_buffer = ExperienceBuffer(
                buffer_size=buffer_size,
                state_dim=self.env.observation_space.shape,
                action_dim=self.env.action_space.shape if hasattr(self.env.action_space, 'shape') else (1,),
                logger=self.logger
            )
        else:
            self.experience_buffer = None
        
        # Lưu cấu hình huấn luyện
        self._save_config()
        
        self.logger.info(f"Đã khởi tạo Trainer cho {agent.name} trên {env.__class__.__name__}")
        self.logger.info(f"Kết quả huấn luyện sẽ được lưu tại {self.output_dir}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Lấy cấu hình mặc định cho huấn luyện.
        
        Returns:
            Dict cấu hình mặc định
        """
        return {
            "num_episodes": 1000,
            "max_steps_per_episode": 1000,
            "eval_frequency": 10,
            "num_eval_episodes": 5,
            "save_frequency": 100,
            "learning_starts": 1000,
            "target_update_frequency": 1000,
            "update_frequency": 4,
            "render_eval": False,
            "render_frequency": 0,  # 0: không render, n: render mỗi n episodes
            "gamma": 0.99,
            "learning_rate": 0.001,
            "epsilon_schedule": {
                "start": 1.0,
                "end": 0.01,
                "decay_steps": 10000
            },
            "batch_size": 64
        }
    
    def _save_config(self) -> None:
        """
        Lưu cấu hình huấn luyện vào file.
        """
        config_path = self.output_dir / "training_config.json"
        
        # Kết hợp config với các thông tin cần thiết khác
        full_config = {
            "experiment_name": self.experiment_name,
            "agent_name": self.agent.name,
            "agent_config": self.agent.config,
            "env_name": self.env.__class__.__name__,
            "training_config": self.config,
            "created_at": datetime.now().isoformat(),
            "additional_params": {
                "use_tensorboard": self.use_tensorboard,
                "save_best_only": self.save_best_only,
                "early_stopping": self.early_stopping,
                "patience": self.patience,
                "verbose": self.verbose,
                **self.kwargs
            }
        }
        
        # Lưu vào file
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(full_config, f, indent=4, ensure_ascii=False)
    
    def _log_to_tensorboard(self, metrics: Dict[str, float], step: int) -> None:
        """
        Ghi metrics vào TensorBoard.
        
        Args:
            metrics: Dict chứa các metrics cần ghi
            step: Bước huấn luyện hiện tại
        """
        if not self.use_tensorboard or self.tensorboard_writer is None:
            return
        
        with self.tensorboard_writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=step)
    
    def train(self) -> Dict[str, List[float]]:
        """
        Huấn luyện agent với cấu hình đã thiết lập.
        
        Returns:
            Dict chứa lịch sử huấn luyện
        """
        # Lấy các tham số từ cấu hình
        num_episodes = self.config["num_episodes"]
        max_steps = self.config["max_steps_per_episode"]
        eval_frequency = self.config["eval_frequency"]
        save_frequency = self.config["save_frequency"]
        render_frequency = self.config["render_frequency"]
        
        self.logger.info(f"Bắt đầu huấn luyện cho {num_episodes} episodes, mỗi episode tối đa {max_steps} bước")
        
        # Thời gian bắt đầu
        start_time = time.time()
        
        # Huấn luyện qua các episodes
        for episode in range(1, num_episodes + 1):
            self.current_episode = episode
            episode_reward, episode_length, episode_losses = self._train_episode(max_steps, render_frequency)
            
            # Cập nhật lịch sử
            self.history["episode_rewards"].append(episode_reward)
            self.history["episode_lengths"].append(episode_length)
            self.history["losses"].append(np.mean(episode_losses) if episode_losses else 0)
            self.history["timestamps"].append(time.time() - start_time)
            
            # Log thông tin
            if self.verbose > 0 and episode % max(1, num_episodes // 100) == 0:
                elapsed_time = time.time() - start_time
                self.logger.info(
                    f"Episode {episode}/{num_episodes} - Reward: {episode_reward:.2f}, "
                    f"Length: {episode_length}, Avg Loss: {self.history['losses'][-1]:.6f}, "
                    f"Elapsed: {elapsed_time:.2f}s"
                )
            
            # Ghi metrics vào TensorBoard
            if self.use_tensorboard:
                metrics = {
                    "training/episode_reward": episode_reward,
                    "training/episode_length": episode_length,
                    "training/avg_loss": self.history["losses"][-1],
                    "training/epsilon": getattr(self.agent, "epsilon", 0)
                }
                self._log_to_tensorboard(metrics, episode)
            
            # Đánh giá định kỳ
            if episode % eval_frequency == 0:
                eval_rewards = self.evaluate()
                mean_eval_reward = np.mean(eval_rewards)
                self.history["eval_rewards"].append(mean_eval_reward)
                
                # Ghi metrics đánh giá vào TensorBoard
                if self.use_tensorboard:
                    metrics = {
                        "evaluation/mean_reward": mean_eval_reward,
                        "evaluation/min_reward": np.min(eval_rewards),
                        "evaluation/max_reward": np.max(eval_rewards)
                    }
                    self._log_to_tensorboard(metrics, episode)
                
                # Lưu mô hình tốt nhất
                if mean_eval_reward > self.best_reward:
                    self.best_reward = mean_eval_reward
                    self.best_episode = episode
                    self.no_improvement_count = 0
                    
                    # Lưu mô hình tốt nhất
                    self.save_agent(is_best=True)
                    
                    if self.verbose > 0:
                        self.logger.info(
                            f"Mô hình mới tốt nhất tại episode {episode} với reward {mean_eval_reward:.2f}"
                        )
                else:
                    self.no_improvement_count += 1
                
                # Kiểm tra early stopping
                if self.early_stopping and self.no_improvement_count >= self.patience:
                    self.logger.info(
                        f"Early stopping tại episode {episode} sau {self.patience} đánh giá không cải thiện. "
                        f"Mô hình tốt nhất tại episode {self.best_episode} với reward {self.best_reward:.2f}"
                    )
                    break
            
            # Lưu mô hình định kỳ
            if episode % save_frequency == 0 and not self.save_best_only:
                self.save_agent(is_best=False, suffix=f"episode_{episode}")
            
            # Tạo checkpoint để tiếp tục huấn luyện nếu bị gián đoạn
            if episode % max(1, num_episodes // 10) == 0:
                self._save_checkpoint()
        
        # Huấn luyện hoàn thành
        self.training_complete = True
        total_time = time.time() - start_time
        
        self.logger.info(
            f"Huấn luyện hoàn thành sau {episode} episodes. "
            f"Thời gian: {total_time:.2f}s. "
            f"Mô hình tốt nhất tại episode {self.best_episode} với reward {self.best_reward:.2f}"
        )
        
        # Lưu mô hình cuối cùng nếu cần
        if not self.save_best_only:
            self.save_agent(is_best=False, suffix="final")
        
        # Lưu lịch sử huấn luyện
        self.save_history()
        
        # Tạo biểu đồ huấn luyện
        self.plot_training_results()
        
        return self.history
    
    def _train_episode(self, max_steps: int, render_frequency: int) -> Tuple[float, int, List[float]]:
        """
        Huấn luyện một episode.
        
        Args:
            max_steps: Số bước tối đa trong episode
            render_frequency: Tần suất render môi trường
            
        Returns:
            Tuple (episode_reward, episode_length, episode_losses)
        """
        # Reset môi trường
        state = self.env.reset()
        episode_reward = 0
        episode_losses = []
        
        # Render môi trường nếu cần
        should_render = render_frequency > 0 and self.current_episode % render_frequency == 0
        if should_render:
            self.env.render()
        
        # Vòng lặp trong một episode
        for step in range(max_steps):
            # Chọn hành động từ agent
            action = self.agent.act(state)
            
            # Thực hiện hành động trong môi trường
            next_state, reward, done, info = self.env.step(action)
            
            # Cập nhật tổng reward
            episode_reward += reward
            
            # Lưu trải nghiệm vào bộ nhớ của agent
            self.agent.remember(state, action, reward, next_state, done)
            
            # Lưu trải nghiệm vào experience buffer nếu có
            if self.experience_buffer is not None:
                self.experience_buffer.add(state, action, reward, next_state, done)
            
            # Học từ bộ nhớ
            loss_info = self.agent.learn()
            if loss_info and 'loss' in loss_info:
                episode_losses.append(loss_info['loss'])
            
            # Cập nhật state
            state = next_state
            
            # Render môi trường nếu cần
            if should_render:
                self.env.render()
            
            # Tăng biến đếm bước
            self.total_steps += 1
            
            # Kết thúc episode nếu done
            if done:
                break
        
        return episode_reward, step + 1, episode_losses
    
    def evaluate(self, num_episodes: Optional[int] = None, render: Optional[bool] = None) -> List[float]:
        """
        Đánh giá agent trong môi trường.
        
        Args:
            num_episodes: Số episodes đánh giá
            render: Có render môi trường không
            
        Returns:
            Danh sách rewards của các episodes
        """
        if num_episodes is None:
            num_episodes = self.config["num_eval_episodes"]
        
        if render is None:
            render = self.config["render_eval"]
        
        if self.verbose > 0:
            self.logger.info(f"Đánh giá agent trên {num_episodes} episodes")
        
        eval_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < self.config["max_steps_per_episode"]:
                # Chọn hành động (không khám phá)
                action = self.agent.act(state, explore=False)
                
                # Thực hiện hành động
                next_state, reward, done, info = self.env.step(action)
                
                # Cập nhật state và reward
                state = next_state
                episode_reward += reward
                step += 1
                
                # Render môi trường nếu cần
                if render:
                    self.env.render()
            
            eval_rewards.append(episode_reward)
            
            if self.verbose > 1:
                self.logger.debug(f"Đánh giá episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}")
        
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        if self.verbose > 0:
            self.logger.info(
                f"Kết quả đánh giá: Mean reward: {mean_reward:.2f} ± {std_reward:.2f}, "
                f"Min: {np.min(eval_rewards):.2f}, Max: {np.max(eval_rewards):.2f}"
            )
        
        return eval_rewards
    
    def save_agent(self, is_best: bool = False, suffix: Optional[str] = None) -> str:
        """
        Lưu mô hình agent.
        
        Args:
            is_best: Có phải là mô hình tốt nhất không
            suffix: Hậu tố cho tên file
            
        Returns:
            Đường dẫn đã lưu
        """
        # Tạo thư mục models
        models_dir = self.output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Xác định tên file
        if is_best:
            model_path = models_dir / "best_model"
        elif suffix:
            model_path = models_dir / f"model_{suffix}"
        else:
            model_path = models_dir / f"model_episode_{self.current_episode}"
        
        # Lưu mô hình
        try:
            self.agent._save_model_impl(model_path)
            
            if self.verbose > 0:
                self.logger.info(f"Đã lưu mô hình tại {model_path}")
            
            return str(model_path)
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu mô hình: {str(e)}")
            return ""
    
    def load_agent(self, model_path: Optional[Union[str, Path]] = None, is_best: bool = False) -> bool:
        """
        Tải mô hình agent.
        
        Args:
            model_path: Đường dẫn tới mô hình
            is_best: Có tải mô hình tốt nhất không
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        if model_path is None:
            models_dir = self.output_dir / "models"
            if is_best:
                model_path = models_dir / "best_model"
            else:
                model_path = models_dir / "final_model"
        
        try:
            success = self.agent._load_model_impl(model_path)
            
            if success and self.verbose > 0:
                self.logger.info(f"Đã tải mô hình từ {model_path}")
            elif not success:
                self.logger.warning(f"Không thể tải mô hình từ {model_path}")
            
            return success
        except Exception as e:
            self.logger.error(f"Lỗi khi tải mô hình: {str(e)}")
            return False
    
    def save_history(self) -> None:
        """
        Lưu lịch sử huấn luyện vào file.
        """
        history_path = self.output_dir / "training_history.json"
        
        # Chuyển đổi numpy arrays thành lists
        serializable_history = {}
        for key, value in self.history.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                serializable_history[key] = [v.tolist() for v in value]
            else:
                serializable_history[key] = value
        
        # Thêm thông tin bổ sung
        serializable_history["total_episodes"] = self.current_episode
        serializable_history["total_steps"] = self.total_steps
        serializable_history["best_episode"] = self.best_episode
        serializable_history["best_reward"] = float(self.best_reward)
        serializable_history["completed"] = self.training_complete
        serializable_history["last_updated"] = datetime.now().isoformat()
        
        # Lưu vào file
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(serializable_history, f, indent=4, ensure_ascii=False)
        
        if self.verbose > 0:
            self.logger.info(f"Đã lưu lịch sử huấn luyện tại {history_path}")
    
    def load_history(self) -> Dict[str, Any]:
        """
        Tải lịch sử huấn luyện từ file.
        
        Returns:
            Dict lịch sử huấn luyện
        """
        history_path = self.output_dir / "training_history.json"
        
        if not history_path.exists():
            self.logger.warning(f"Không tìm thấy file lịch sử tại {history_path}")
            return {}
        
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            
            # Cập nhật lịch sử hiện tại
            for key in self.history.keys():
                if key in history:
                    self.history[key] = history[key]
            
            # Cập nhật các biến trạng thái
            if "total_episodes" in history:
                self.current_episode = history["total_episodes"]
            
            if "total_steps" in history:
                self.total_steps = history["total_steps"]
            
            if "best_episode" in history:
                self.best_episode = history["best_episode"]
            
            if "best_reward" in history:
                self.best_reward = history["best_reward"]
            
            if "completed" in history:
                self.training_complete = history["completed"]
            
            self.logger.info(f"Đã tải lịch sử huấn luyện từ {history_path}")
            
            return history
        except Exception as e:
            self.logger.error(f"Lỗi khi tải lịch sử huấn luyện: {str(e)}")
            return {}
    
    def _save_checkpoint(self) -> None:
        """
        Lưu checkpoint để có thể tiếp tục huấn luyện sau này.
        """
        # Tạo thư mục checkpoints
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Lưu trạng thái agent
        checkpoint_path = checkpoint_dir / f"checkpoint_episode_{self.current_episode}"
        self.agent._save_model_impl(checkpoint_path)
        
        # Lưu lịch sử huấn luyện
        self.save_history()
        
        if self.verbose > 1:
            self.logger.debug(f"Đã lưu checkpoint tại episode {self.current_episode}")
    
    def restore_checkpoint(self, episode: Optional[int] = None) -> bool:
        """
        Khôi phục từ checkpoint.
        
        Args:
            episode: Số episode của checkpoint (None để dùng gần nhất)
            
        Returns:
            True nếu khôi phục thành công, False nếu không
        """
        checkpoint_dir = self.output_dir / "checkpoints"
        
        if not checkpoint_dir.exists():
            self.logger.warning(f"Không tìm thấy thư mục checkpoints tại {checkpoint_dir}")
            return False
        
        # Tìm checkpoint
        if episode is not None:
            checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode}"
            if not checkpoint_path.exists():
                self.logger.warning(f"Không tìm thấy checkpoint cho episode {episode}")
                return False
        else:
            # Tìm checkpoint gần nhất
            checkpoints = list(checkpoint_dir.glob("checkpoint_episode_*"))
            if not checkpoints:
                self.logger.warning("Không tìm thấy checkpoints nào")
                return False
            
            # Sắp xếp theo số episode giảm dần
            checkpoints.sort(key=lambda x: int(x.name.split("_")[-1]), reverse=True)
            checkpoint_path = checkpoints[0]
            episode = int(checkpoint_path.name.split("_")[-1])
        
        # Tải checkpoint
        success = self.agent._load_model_impl(checkpoint_path)
        
        if success:
            self.logger.info(f"Đã khôi phục checkpoint từ episode {episode}")
            
            # Tải lịch sử
            self.load_history()
            
            # Cập nhật biến trạng thái nếu cần
            if self.current_episode < episode:
                self.current_episode = episode
            
            return True
        else:
            self.logger.error(f"Không thể tải checkpoint từ {checkpoint_path}")
            return False
    
    def plot_training_results(self) -> None:
        """
        Tạo biểu đồ kết quả huấn luyện.
        """
        # Kiểm tra xem có đủ dữ liệu không
        if not self.history["episode_rewards"]:
            self.logger.warning("Không đủ dữ liệu để tạo biểu đồ")
            return
        
        # Tạo thư mục plots
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Tạo hình với 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Episode Rewards
        axs[0, 0].plot(self.history["episode_rewards"])
        axs[0, 0].set_title("Episode Rewards")
        axs[0, 0].set_xlabel("Episode")
        axs[0, 0].set_ylabel("Reward")
        axs[0, 0].grid(True)
        
        # Thêm đường trung bình chạy
        if len(self.history["episode_rewards"]) > 10:
            window_size = min(100, len(self.history["episode_rewards"]) // 10)
            running_avg = np.convolve(
                self.history["episode_rewards"], 
                np.ones(window_size) / window_size, 
                mode='valid'
            )
            axs[0, 0].plot(
                range(window_size - 1, window_size - 1 + len(running_avg)), 
                running_avg, 
                'r-', 
                linewidth=2
            )
            axs[0, 0].legend(["Rewards", f"Running Avg ({window_size} ep)"])
        
        # 2. Episode Lengths
        axs[0, 1].plot(self.history["episode_lengths"])
        axs[0, 1].set_title("Episode Lengths")
        axs[0, 1].set_xlabel("Episode")
        axs[0, 1].set_ylabel("Steps")
        axs[0, 1].grid(True)
        
        # 3. Training Losses
        if self.history["losses"]:
            axs[1, 0].plot(self.history["losses"])
            axs[1, 0].set_title("Training Loss")
            axs[1, 0].set_xlabel("Episode")
            axs[1, 0].set_ylabel("Loss")
            axs[1, 0].grid(True)
            
            # Sử dụng log scale nếu có sự khác biệt lớn
            if max(self.history["losses"]) > 10 * min(filter(lambda x: x > 0, self.history["losses"])):
                axs[1, 0].set_yscale('log')
        
        # 4. Evaluation Rewards
        if self.history["eval_rewards"]:
            # Tính toán các episodes đánh giá
            eval_episodes = range(
                self.config["eval_frequency"], 
                self.config["eval_frequency"] * (len(self.history["eval_rewards"]) + 1), 
                self.config["eval_frequency"]
            )
            
            axs[1, 1].plot(eval_episodes, self.history["eval_rewards"], 'go-')
            axs[1, 1].set_title("Evaluation Rewards")
            axs[1, 1].set_xlabel("Episode")
            axs[1, 1].set_ylabel("Average Reward")
            axs[1, 1].grid(True)
            
            # Đánh dấu mô hình tốt nhất
            if self.best_episode > 0:
                axs[1, 1].scatter(
                    [self.best_episode], 
                    [self.best_reward], 
                    c='r', 
                    marker='*', 
                    s=200, 
                    label=f"Best Model (ep {self.best_episode})"
                )
                axs[1, 1].legend()
        
        # Tối ưu hóa layout
        plt.tight_layout()
        
        # Thêm tiêu đề chính
        plt.suptitle(
            f"Training Results: {self.experiment_name}\n"
            f"Agent: {self.agent.name}, Environment: {self.env.__class__.__name__}",
            fontsize=16
        )
        
        # Điều chỉnh spacing
        plt.subplots_adjust(top=0.9)
        
        # Lưu biểu đồ
        plot_path = plots_dir / "training_results.png"
        plt.savefig(plot_path)
        plt.close(fig)
        
        if self.verbose > 0:
            self.logger.info(f"Đã lưu biểu đồ kết quả huấn luyện tại {plot_path}")
    
    def export_report(self) -> str:
        """
        Xuất báo cáo huấn luyện dạng Markdown.
        
        Returns:
            Đường dẫn file báo cáo
        """
        report_path = self.output_dir / "training_report.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            # Tiêu đề
            f.write(f"# Báo cáo Huấn luyện: {self.experiment_name}\n\n")
            
            # Thông tin chung
            f.write("## Thông tin chung\n\n")
            f.write(f"- **Thời gian:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Agent:** {self.agent.name}\n")
            f.write(f"- **Môi trường:** {self.env.__class__.__name__}\n")
            f.write(f"- **Số episodes huấn luyện:** {self.current_episode}\n")
            f.write(f"- **Tổng số bước:** {self.total_steps}\n")
            f.write(f"- **Trạng thái:** {'Hoàn thành' if self.training_complete else 'Đang huấn luyện'}\n\n")
            
            # Kết quả huấn luyện
            f.write("## Kết quả huấn luyện\n\n")
            
            if self.history["episode_rewards"]:
                # Tính toán thông tin thống kê
                recent_rewards = self.history["episode_rewards"][-10:]
                
                f.write(f"- **Mô hình tốt nhất tại episode:** {self.best_episode}\n")
                f.write(f"- **Phần thưởng đánh giá tốt nhất:** {self.best_reward:.2f}\n")
                f.write(f"- **Phần thưởng trung bình (10 episodes cuối):** {np.mean(recent_rewards):.2f}\n")
                f.write(f"- **Phần thưởng cao nhất:** {np.max(self.history['episode_rewards']):.2f}\n")
                f.write(f"- **Phần thưởng thấp nhất:** {np.min(self.history['episode_rewards']):.2f}\n")
                
                if self.history["eval_rewards"]:
                    f.write(f"- **Phần thưởng đánh giá cuối cùng:** {self.history['eval_rewards'][-1]:.2f}\n")
            else:
                f.write("*Chưa có dữ liệu huấn luyện*\n")
            
            f.write("\n")
            
            # Cấu hình huấn luyện
            f.write("## Cấu hình huấn luyện\n\n")
            f.write("```json\n")
            json.dump(self.config, f, indent=2, ensure_ascii=False)
            f.write("\n```\n\n")
            
            # Cấu hình agent
            f.write("## Cấu hình agent\n\n")
            f.write("```json\n")
            json.dump(self.agent.config, f, indent=2, ensure_ascii=False)
            f.write("\n```\n\n")
            
            # Biểu đồ
            if (self.output_dir / "plots" / "training_results.png").exists():
                f.write("## Biểu đồ huấn luyện\n\n")
                f.write("![Training Results](plots/training_results.png)\n\n")
            
            # Kết luận
            f.write("## Kết luận\n\n")
            
            if self.training_complete:
                if self.best_reward > 0:
                    f.write("Quá trình huấn luyện đã hoàn thành thành công. ")
                    f.write(f"Mô hình tốt nhất đạt được phần thưởng đánh giá {self.best_reward:.2f} tại episode {self.best_episode}.\n\n")
                else:
                    f.write("Quá trình huấn luyện đã hoàn thành nhưng kết quả chưa đạt kỳ vọng. ")
                    f.write("Cân nhắc điều chỉnh cấu hình và thử lại.\n\n")
                    
                if self.early_stopping and self.no_improvement_count >= self.patience:
                    f.write(f"Huấn luyện kết thúc sớm sau {self.no_improvement_count} lần đánh giá không có cải thiện.\n\n")
            else:
                f.write("Quá trình huấn luyện vẫn đang tiếp tục. Báo cáo này được tạo làm mốc trung gian.\n\n")
        
        if self.verbose > 0:
            self.logger.info(f"Đã xuất báo cáo huấn luyện tại {report_path}")
        
        return str(report_path)