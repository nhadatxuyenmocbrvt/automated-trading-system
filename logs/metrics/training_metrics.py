"""
Đo lường hiệu suất huấn luyện.
File này cung cấp các công cụ để theo dõi, ghi lại, và phân tích các
số liệu trong quá trình huấn luyện agent, giúp đánh giá hiệu suất
và đưa ra quyết định tối ưu.
"""

import os
import time
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path

# Import các module từ hệ thống
from config.logging_config import get_logger, log_training_metrics
from config.system_config import get_system_config
import tensorflow as tf

class TrainingMetricsTracker:
    """
    Lớp theo dõi và quản lý các số liệu trong quá trình huấn luyện.
    Cung cấp các phương thức để ghi nhận, phân tích, và xuất báo cáo
    về hiệu suất huấn luyện.
    """
    
    def __init__(
        self,
        experiment_name: str,
        agent_name: str,
        env_name: str,
        output_dir: Optional[Union[str, Path]] = None,
        use_tensorboard: bool = True,
        use_csv: bool = True,
        log_frequency: int = 1,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Khởi tạo theo dõi số liệu huấn luyện.
        
        Args:
            experiment_name: Tên thí nghiệm
            agent_name: Tên agent đang huấn luyện
            env_name: Tên môi trường huấn luyện
            output_dir: Thư mục đầu ra cho số liệu
            use_tensorboard: Sử dụng TensorBoard để theo dõi số liệu
            use_csv: Lưu số liệu vào file CSV
            log_frequency: Tần suất ghi log (mỗi bao nhiêu episode)
            logger: Logger tùy chỉnh
            config: Cấu hình bổ sung
        """
        # Thiết lập logger
        self.logger = logger or get_logger("training_metrics")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập thông tin cơ bản
        self.experiment_name = experiment_name
        self.agent_name = agent_name
        self.env_name = env_name
        self.use_tensorboard = use_tensorboard
        self.use_csv = use_csv
        self.log_frequency = log_frequency
        
        # Cấu hình theo dõi
        self.config = config or {}
        
        # Thiết lập thư mục đầu ra
        if output_dir is None:
            logs_dir = Path(self.system_config.get("log_dir", "./logs"))
            output_dir = logs_dir / "training" / self.experiment_name
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập file CSV
        if self.use_csv:
            self.csv_path = self.output_dir / f"{self.experiment_name}_metrics.csv"
            
            # Kiểm tra file có tồn tại không
            if not self.csv_path.exists():
                # Tạo DataFrame rỗng với các cột dữ liệu chính
                df = pd.DataFrame(columns=[
                    'timestamp', 'episode', 'step', 'reward', 'loss', 
                    'episode_length', 'epsilon', 'entropy', 'kl_divergence',
                    'win_rate', 'training_time', 'learning_rate'
                ])
                df.to_csv(self.csv_path, index=False)
                self.logger.info(f"Đã tạo file CSV mới tại {self.csv_path}")
        
        # Thiết lập TensorBoard
        if self.use_tensorboard:
            self.tensorboard_dir = self.output_dir / "tensorboard"
            self.tensorboard_dir.mkdir(exist_ok=True)
            self.summary_writer = tf.summary.create_file_writer(str(self.tensorboard_dir))
            self.logger.info(f"Đã thiết lập TensorBoard tại {self.tensorboard_dir}")
        else:
            self.summary_writer = None
        
        # Khởi tạo biến trạng thái
        self.current_episode = 0
        self.total_steps = 0
        self.start_time = time.time()
        self.episode_start_time = time.time()
        
        # Theo dõi số liệu
        self.metrics_history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": [],
            "epsilons": [],
            "entropies": [],
            "kl_divergences": [],
            "win_rates": [],
            "episode_times": [],
            "learning_rates": [],
            "timestamps": []
        }
        
        # Các số liệu cho episode hiện tại
        self.current_metrics = {
            "episode_reward": 0.0,
            "episode_length": 0,
            "losses": [],
            "win_count": 0,
            "loss_count": 0
        }
        
        self.logger.info(f"Đã khởi tạo TrainingMetricsTracker cho {agent_name} trên {env_name}")
    
    def start_episode(self, episode: int) -> None:
        """
        Bắt đầu theo dõi một episode mới.
        
        Args:
            episode: Số thứ tự episode
        """
        self.current_episode = episode
        self.episode_start_time = time.time()
        
        # Reset số liệu cho episode mới
        self.current_metrics = {
            "episode_reward": 0.0,
            "episode_length": 0,
            "losses": [],
            "win_count": 0,
            "loss_count": 0
        }
        
        self.logger.debug(f"Bắt đầu episode {episode}")
    
    def log_step(
        self,
        reward: float,
        loss: Optional[float] = None,
        epsilon: Optional[float] = None,
        action: Optional[int] = None,
        state: Optional[np.ndarray] = None,
        next_state: Optional[np.ndarray] = None,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Ghi nhận thông tin một bước huấn luyện.
        
        Args:
            reward: Phần thưởng cho bước hiện tại
            loss: Giá trị loss nếu có học tập trong bước này
            epsilon: Giá trị epsilon cho chiến lược ε-greedy
            action: Hành động đã thực hiện
            state: Trạng thái hiện tại
            next_state: Trạng thái kế tiếp
            done: Episode đã kết thúc hay chưa
            info: Thông tin bổ sung từ môi trường
        """
        # Cập nhật số liệu cho episode hiện tại
        self.current_metrics["episode_reward"] += reward
        self.current_metrics["episode_length"] += 1
        
        # Theo dõi win/loss nếu có thông tin
        if info and "win" in info:
            if info["win"]:
                self.current_metrics["win_count"] += 1
            else:
                self.current_metrics["loss_count"] += 1
        
        # Lưu loss nếu có
        if loss is not None:
            self.current_metrics["losses"].append(loss)
        
        # Tăng biến đếm bước
        self.total_steps += 1
    
    def end_episode(
        self,
        epsilon: Optional[float] = None,
        entropy: Optional[float] = None,
        kl_divergence: Optional[float] = None,
        learning_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Kết thúc và tổng kết một episode huấn luyện.
        
        Args:
            epsilon: Giá trị epsilon hiện tại
            entropy: Giá trị entropy của policy
            kl_divergence: Giá trị KL divergence
            learning_rate: Tốc độ học hiện tại
            
        Returns:
            Dict với các số liệu của episode
        """
        # Thời gian huấn luyện episode
        episode_time = time.time() - self.episode_start_time
        
        # Tính giá trị trung bình của loss
        avg_loss = np.mean(self.current_metrics["losses"]) if self.current_metrics["losses"] else None
        
        # Tính win rate
        total_games = self.current_metrics["win_count"] + self.current_metrics["loss_count"]
        win_rate = self.current_metrics["win_count"] / total_games if total_games > 0 else None
        
        # Tạo bản ghi số liệu
        metrics = {
            "episode": self.current_episode,
            "reward": self.current_metrics["episode_reward"],
            "length": self.current_metrics["episode_length"],
            "loss": avg_loss,
            "epsilon": epsilon,
            "entropy": entropy,
            "kl_divergence": kl_divergence,
            "win_rate": win_rate,
            "episode_time": episode_time,
            "learning_rate": learning_rate,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cập nhật lịch sử số liệu
        self.metrics_history["episode_rewards"].append(self.current_metrics["episode_reward"])
        self.metrics_history["episode_lengths"].append(self.current_metrics["episode_length"])
        self.metrics_history["losses"].append(avg_loss if avg_loss is not None else 0.0)
        self.metrics_history["epsilons"].append(epsilon if epsilon is not None else 0.0)
        self.metrics_history["entropies"].append(entropy if entropy is not None else 0.0)
        self.metrics_history["kl_divergences"].append(kl_divergence if kl_divergence is not None else 0.0)
        self.metrics_history["win_rates"].append(win_rate if win_rate is not None else 0.0)
        self.metrics_history["episode_times"].append(episode_time)
        self.metrics_history["learning_rates"].append(learning_rate if learning_rate is not None else 0.0)
        self.metrics_history["timestamps"].append(datetime.now().isoformat())
        
        # Ghi log nếu cần
        if self.current_episode % self.log_frequency == 0:
            log_training_metrics(
                self.logger,
                episode=self.current_episode,
                reward=self.current_metrics["episode_reward"],
                winrate=win_rate if win_rate is not None else 0.0,
                loss=avg_loss if avg_loss is not None else 0.0,
                kl=kl_divergence if kl_divergence is not None else 0.0,
                entropy=entropy if entropy is not None else 0.0
            )
        
        # Ghi vào TensorBoard
        if self.use_tensorboard and self.summary_writer is not None:
            self._write_to_tensorboard(metrics)
        
        # Ghi vào CSV
        if self.use_csv:
            self._append_to_csv(metrics)
        
        # Trả về số liệu
        return metrics
    
    def _write_to_tensorboard(self, metrics: Dict[str, Any]) -> None:
        """
        Ghi số liệu vào TensorBoard.
        
        Args:
            metrics: Dict chứa các số liệu cần ghi
        """
        try:
            with self.summary_writer.as_default():
                # Số liệu cơ bản
                if "reward" in metrics and metrics["reward"] is not None:
                    tf.summary.scalar("Reward/episode", metrics["reward"], step=metrics["episode"])
                
                if "loss" in metrics and metrics["loss"] is not None:
                    tf.summary.scalar("Loss/episode", metrics["loss"], step=metrics["episode"])
                
                if "length" in metrics and metrics["length"] is not None:
                    tf.summary.scalar("Length/episode", metrics["length"], step=metrics["episode"])
                
                # Số liệu khám phá và học tập
                if "epsilon" in metrics and metrics["epsilon"] is not None:
                    tf.summary.scalar("Epsilon/episode", metrics["epsilon"], step=metrics["episode"])
                
                if "entropy" in metrics and metrics["entropy"] is not None:
                    tf.summary.scalar("Entropy/episode", metrics["entropy"], step=metrics["episode"])
                
                if "kl_divergence" in metrics and metrics["kl_divergence"] is not None:
                    tf.summary.scalar("KL_Divergence/episode", metrics["kl_divergence"], step=metrics["episode"])
                
                # Số liệu hiệu suất
                if "win_rate" in metrics and metrics["win_rate"] is not None:
                    tf.summary.scalar("Win_Rate/episode", metrics["win_rate"], step=metrics["episode"])
                
                if "episode_time" in metrics and metrics["episode_time"] is not None:
                    tf.summary.scalar("Time/episode", metrics["episode_time"], step=metrics["episode"])
                
                if "learning_rate" in metrics and metrics["learning_rate"] is not None:
                    tf.summary.scalar("Learning_Rate/episode", metrics["learning_rate"], step=metrics["episode"])
                
                # Flush để đảm bảo ghi ngay lập tức
                self.summary_writer.flush()
        except Exception as e:
            self.logger.warning(f"Không thể ghi số liệu vào TensorBoard: {str(e)}")
    
    def _append_to_csv(self, metrics: Dict[str, Any]) -> None:
        """
        Thêm số liệu vào file CSV.
        
        Args:
            metrics: Dict chứa các số liệu cần ghi
        """
        try:
            # Tạo DataFrame mới từ metrics
            new_data = pd.DataFrame([{
                'timestamp': datetime.now().isoformat(),
                'episode': metrics.get('episode'),
                'step': self.total_steps,
                'reward': metrics.get('reward'),
                'loss': metrics.get('loss'),
                'episode_length': metrics.get('length'),
                'epsilon': metrics.get('epsilon'),
                'entropy': metrics.get('entropy'),
                'kl_divergence': metrics.get('kl_divergence'),
                'win_rate': metrics.get('win_rate'),
                'training_time': metrics.get('episode_time'),
                'learning_rate': metrics.get('learning_rate')
            }])
            
            # Thêm vào file CSV
            new_data.to_csv(self.csv_path, mode='a', header=False, index=False)
            
        except Exception as e:
            self.logger.warning(f"Không thể ghi số liệu vào CSV: {str(e)}")
    
    def save_metrics(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Lưu lịch sử số liệu vào file JSON.
        
        Args:
            path: Đường dẫn file (None để tạo tự động)
            
        Returns:
            Đường dẫn file đã lưu
        """
        # Tạo đường dẫn tự động nếu không được cung cấp
        if path is None:
            path = self.output_dir / f"{self.experiment_name}_metrics.json"
        else:
            path = Path(path)
        
        # Chuẩn bị dữ liệu để lưu
        save_data = {
            "experiment_name": self.experiment_name,
            "agent_name": self.agent_name,
            "env_name": self.env_name,
            "total_episodes": self.current_episode,
            "total_steps": self.total_steps,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "training_duration": time.time() - self.start_time,
            "metrics_history": {}
        }
        
        # Chuyển đổi np.array thành list
        for key, values in self.metrics_history.items():
            if isinstance(values, list) and len(values) > 0:
                if isinstance(values[0], np.ndarray):
                    save_data["metrics_history"][key] = [v.tolist() for v in values]
                elif isinstance(values[0], (np.int32, np.int64, np.float32, np.float64)):
                    save_data["metrics_history"][key] = [float(v) for v in values]
                else:
                    save_data["metrics_history"][key] = values
            else:
                save_data["metrics_history"][key] = values
        
        # Thêm thống kê tóm tắt
        if len(self.metrics_history["episode_rewards"]) > 0:
            rewards = self.metrics_history["episode_rewards"]
            save_data["summary"] = {
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "min_reward": float(np.min(rewards)),
                "max_reward": float(np.max(rewards)),
                "median_reward": float(np.median(rewards)),
                "final_reward": float(rewards[-1]),
                "mean_last_10": float(np.mean(rewards[-10:])),
                "mean_last_100": float(np.mean(rewards[-100:])) if len(rewards) >= 100 else None
            }
            
            # Win rate
            win_rates = [wr for wr in self.metrics_history["win_rates"] if wr is not None]
            if win_rates:
                save_data["summary"]["mean_win_rate"] = float(np.mean(win_rates))
                save_data["summary"]["final_win_rate"] = float(win_rates[-1])
        
        # Lưu vào file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"Đã lưu số liệu huấn luyện vào {path}")
        return str(path)
    
    def load_metrics(self, path: Union[str, Path]) -> bool:
        """
        Tải số liệu từ file JSON.
        
        Args:
            path: Đường dẫn file
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            path = Path(path)
            
            # Đọc file
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Cập nhật thông tin
            self.experiment_name = data.get("experiment_name", self.experiment_name)
            self.agent_name = data.get("agent_name", self.agent_name)
            self.env_name = data.get("env_name", self.env_name)
            self.current_episode = data.get("total_episodes", 0)
            self.total_steps = data.get("total_steps", 0)
            
            # Cập nhật lịch sử số liệu
            if "metrics_history" in data:
                # Chỉ cập nhật các key có trong cả hai
                for key in self.metrics_history.keys():
                    if key in data["metrics_history"]:
                        self.metrics_history[key] = data["metrics_history"][key]
            
            self.logger.info(f"Đã tải số liệu từ {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Không thể tải số liệu từ {path}: {str(e)}")
            return False
    
    def plot_metrics(self, save_path: Optional[Union[str, Path]] = None) -> List[Path]:
        """
        Tạo và lưu biểu đồ số liệu huấn luyện.
        
        Args:
            save_path: Thư mục lưu biểu đồ (None để sử dụng output_dir)
            
        Returns:
            Danh sách đường dẫn các file biểu đồ đã lưu
        """
        # Kiểm tra nếu không có dữ liệu
        if len(self.metrics_history["episode_rewards"]) == 0:
            self.logger.warning("Không có dữ liệu để tạo biểu đồ")
            return []
        
        # Thiết lập thư mục lưu
        if save_path is None:
            save_path = self.output_dir / "plots"
        else:
            save_path = Path(save_path)
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Danh sách để lưu đường dẫn các file
        saved_paths = []
        
        # ---------- 1. Biểu đồ phần thưởng ----------
        try:
            plt.figure(figsize=(12, 6))
            rewards = self.metrics_history["episode_rewards"]
            episodes = list(range(1, len(rewards) + 1))
            plt.plot(episodes, rewards, 'b-', label='Episode Reward')
            
            # Thêm đường trung bình trượt
            if len(rewards) > 10:
                window_size = min(100, len(rewards) // 10)
                rolling_mean = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
                plt.plot(range(window_size, window_size + len(rolling_mean)), rolling_mean, 'r-', 
                         label=f'Trung bình trượt ({window_size} episodes)')
            
            plt.grid(True, alpha=0.3)
            plt.title('Phần thưởng theo episode')
            plt.xlabel('Episode')
            plt.ylabel('Phần thưởng')
            plt.legend()
            
            # Lưu biểu đồ
            reward_path = save_path / f"{self.experiment_name}_rewards.png"
            plt.savefig(reward_path)
            plt.close()
            saved_paths.append(reward_path)
            
        except Exception as e:
            self.logger.warning(f"Không thể tạo biểu đồ phần thưởng: {str(e)}")
        
        # ---------- 2. Biểu đồ Loss ----------
        try:
            losses = self.metrics_history["losses"]
            if len(losses) > 0 and not all(l == 0 for l in losses):
                plt.figure(figsize=(12, 6))
                episodes = list(range(1, len(losses) + 1))
                plt.plot(episodes, losses, 'g-', label='Loss')
                
                # Thêm đường trung bình trượt
                if len(losses) > 10:
                    window_size = min(100, len(losses) // 10)
                    rolling_mean = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')
                    plt.plot(range(window_size, window_size + len(rolling_mean)), rolling_mean, 'r-', 
                             label=f'Trung bình trượt ({window_size} episodes)')
                
                plt.grid(True, alpha=0.3)
                plt.title('Loss theo episode')
                plt.xlabel('Episode')
                plt.ylabel('Loss')
                plt.legend()
                
                # Sử dụng log scale nếu giá trị loss có sự chênh lệch lớn
                if max(losses) / (min(filter(lambda x: x > 0, losses)) + 1e-10) > 100:
                    plt.yscale('log')
                
                # Lưu biểu đồ
                loss_path = save_path / f"{self.experiment_name}_losses.png"
                plt.savefig(loss_path)
                plt.close()
                saved_paths.append(loss_path)
                
        except Exception as e:
            self.logger.warning(f"Không thể tạo biểu đồ Loss: {str(e)}")
        
        # ---------- 3. Biều đồ Win rate ----------
        try:
            win_rates = self.metrics_history["win_rates"]
            # Lọc các giá trị None
            win_rates = [wr for wr in win_rates if wr is not None]
            
            if len(win_rates) > 0:
                plt.figure(figsize=(12, 6))
                episodes = list(range(1, len(win_rates) + 1))
                plt.plot(episodes, win_rates, 'r-', label='Win Rate')
                
                # Thêm đường trung bình trượt
                if len(win_rates) > 10:
                    window_size = min(100, len(win_rates) // 10)
                    rolling_mean = np.convolve(win_rates, np.ones(window_size) / window_size, mode='valid')
                    plt.plot(range(window_size, window_size + len(rolling_mean)), rolling_mean, 'g-', 
                             label=f'Trung bình trượt ({window_size} episodes)')
                
                plt.grid(True, alpha=0.3)
                plt.title('Tỷ lệ thắng theo episode')
                plt.xlabel('Episode')
                plt.ylabel('Win Rate')
                plt.ylim([0, 1])
                plt.legend()
                
                # Lưu biểu đồ
                win_rate_path = save_path / f"{self.experiment_name}_win_rates.png"
                plt.savefig(win_rate_path)
                plt.close()
                saved_paths.append(win_rate_path)
                
        except Exception as e:
            self.logger.warning(f"Không thể tạo biểu đồ Win rate: {str(e)}")
        
        # ---------- 4. Biểu đồ tổng hợp ----------
        try:
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            
            # Biểu đồ phần thưởng
            axs[0, 0].plot(episodes, rewards, 'b-')
            axs[0, 0].set_title('Phần thưởng')
            axs[0, 0].set_xlabel('Episode')
            axs[0, 0].set_ylabel('Reward')
            axs[0, 0].grid(True, alpha=0.3)
            
            # Biểu đồ Loss
            if len(losses) > 0 and not all(l == 0 for l in losses):
                axs[0, 1].plot(range(1, len(losses) + 1), losses, 'g-')
                axs[0, 1].set_title('Loss')
                axs[0, 1].set_xlabel('Episode')
                axs[0, 1].set_ylabel('Loss')
                axs[0, 1].grid(True, alpha=0.3)
                
                # Sử dụng log scale nếu giá trị loss có sự chênh lệch lớn
                if max(losses) / (min(filter(lambda x: x > 0, losses)) + 1e-10) > 100:
                    axs[0, 1].set_yscale('log')
            
            # Biểu đồ Win rate
            if len(win_rates) > 0:
                axs[1, 0].plot(range(1, len(win_rates) + 1), win_rates, 'r-')
                axs[1, 0].set_title('Tỷ lệ thắng')
                axs[1, 0].set_xlabel('Episode')
                axs[1, 0].set_ylabel('Win Rate')
                axs[1, 0].set_ylim([0, 1])
                axs[1, 0].grid(True, alpha=0.3)
            
            # Biểu đồ Epsilon
            epsilons = self.metrics_history["epsilons"]
            if len(epsilons) > 0 and not all(e == 0 for e in epsilons):
                axs[1, 1].plot(range(1, len(epsilons) + 1), epsilons, 'c-')
                axs[1, 1].set_title('Epsilon')
                axs[1, 1].set_xlabel('Episode')
                axs[1, 1].set_ylabel('Epsilon')
                axs[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Lưu biểu đồ
            summary_path = save_path / f"{self.experiment_name}_summary.png"
            plt.savefig(summary_path)
            plt.close()
            saved_paths.append(summary_path)
            
        except Exception as e:
            self.logger.warning(f"Không thể tạo biểu đồ tổng hợp: {str(e)}")
        
        return saved_paths
    
    def generate_training_report(self, save_path: Optional[Union[str, Path]] = None) -> str:
        """
        Tạo báo cáo Markdown về quá trình huấn luyện.
        
        Args:
            save_path: Đường dẫn file báo cáo (None để tạo tự động)
            
        Returns:
            Đường dẫn file báo cáo
        """
        # Thiết lập đường dẫn lưu
        if save_path is None:
            save_path = self.output_dir / f"{self.experiment_name}_report.md"
        else:
            save_path = Path(save_path)
        
        # Tạo thư mục nếu chưa tồn tại
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Tạo biểu đồ và lưu
        plot_paths = self.plot_metrics()
        
        # Chuẩn bị dữ liệu cho báo cáo
        rewards = self.metrics_history["episode_rewards"]
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Tạo báo cáo
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"# Báo cáo huấn luyện: {self.experiment_name}\n\n")
            
            f.write("## Thông tin chung\n\n")
            f.write(f"- **Thời gian báo cáo:** {current_time}\n")
            f.write(f"- **Tên agent:** {self.agent_name}\n")
            f.write(f"- **Môi trường:** {self.env_name}\n")
            f.write(f"- **Tổng số episode:** {self.current_episode}\n")
            f.write(f"- **Tổng số bước:** {self.total_steps}\n")
            
            total_time = time.time() - self.start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            f.write(f"- **Thời gian huấn luyện:** {int(hours)} giờ {int(minutes)} phút {int(seconds)} giây\n\n")
            
            if len(rewards) > 0:
                f.write("## Thống kê hiệu suất\n\n")
                f.write(f"- **Phần thưởng trung bình:** {np.mean(rewards):.2f}\n")
                f.write(f"- **Phần thưởng cao nhất:** {np.max(rewards):.2f}\n")
                f.write(f"- **Phần thưởng thấp nhất:** {np.min(rewards):.2f}\n")
                f.write(f"- **Độ lệch chuẩn phần thưởng:** {np.std(rewards):.2f}\n")
                
                if len(rewards) >= 10:
                    f.write(f"- **Phần thưởng trung bình (10 episode cuối):** {np.mean(rewards[-10:]):.2f}\n")
                
                if len(rewards) >= 100:
                    f.write(f"- **Phần thưởng trung bình (100 episode cuối):** {np.mean(rewards[-100:]):.2f}\n")
                
                # Thêm win rate nếu có
                win_rates = [wr for wr in self.metrics_history["win_rates"] if wr is not None]
                if win_rates:
                    f.write(f"- **Tỷ lệ thắng trung bình:** {np.mean(win_rates):.2%}\n")
                    f.write(f"- **Tỷ lệ thắng cuối cùng:** {win_rates[-1]:.2%}\n")
                
                f.write("\n")
            
            # Hiển thị biểu đồ
            f.write("## Biểu đồ\n\n")
            
            for plot_path in plot_paths:
                rel_path = os.path.relpath(plot_path, save_path.parent)
                plot_name = os.path.splitext(os.path.basename(plot_path))[0].replace(f"{self.experiment_name}_", "")
                f.write(f"### {plot_name.capitalize()}\n\n")
                f.write(f"![{plot_name}]({rel_path})\n\n")
            
            # Thêm phần kết luận
            f.write("## Kết luận\n\n")
            
            # Đánh giá quá trình huấn luyện
            if len(rewards) >= 10:
                avg_first_10 = np.mean(rewards[:10])
                avg_last_10 = np.mean(rewards[-10:])
                improvement = avg_last_10 - avg_first_10
                
                if improvement > 0:
                    f.write(f"Quá trình huấn luyện đã cải thiện hiệu suất agent. Phần thưởng trung bình đã tăng từ {avg_first_10:.2f} (10 episode đầu) lên {avg_last_10:.2f} (10 episode cuối), tương đương với sự cải thiện {improvement:.2f} ({improvement/max(1, abs(avg_first_10))*100:.2f}%).\n\n")
                else:
                    f.write(f"Quá trình huấn luyện chưa cải thiện được hiệu suất agent. Phần thưởng trung bình đã thay đổi từ {avg_first_10:.2f} (10 episode đầu) thành {avg_last_10:.2f} (10 episode cuối), tương đương với sự suy giảm {-improvement:.2f} ({-improvement/max(1, abs(avg_first_10))*100:.2f}%).\n\n")
            
            # Thêm đề xuất tiếp theo
            f.write("### Đề xuất tiếp theo\n\n")
            
            if len(rewards) >= 10:
                # Kiểm tra sự hội tụ
                last_10_std = np.std(rewards[-10:])
                last_half_mean = np.mean(rewards[len(rewards)//2:])
                last_10_mean = np.mean(rewards[-10:])
                
                if last_10_std < 0.1 * abs(last_10_mean):
                    f.write("- Agent dường như đã hội tụ (độ lệch chuẩn của 10 episode cuối thấp). Có thể cân nhắc dừng huấn luyện hoặc giảm learning rate để tinh chỉnh.\n")
                
                if last_10_mean < last_half_mean:
                    f.write("- Hiệu suất có dấu hiệu suy giảm trong các episode gần đây. Cân nhắc giảm learning rate hoặc kiểm tra lại cấu trúc mạng.\n")
                
                if np.mean(rewards[-20:-10]) > last_10_mean:
                    f.write("- Hiệu suất có xu hướng giảm trong 10 episode cuối cùng. Có thể agent đang bị overfit, cân nhắc áp dụng regularization.\n")
            
            f.write("- Thử nghiệm với các kết hợp hyperparameter khác để cải thiện hiệu suất.\n")
            f.write("- Cân nhắc điều chỉnh cấu trúc reward function để khuyến khích hành vi mong muốn.\n")
            f.write("- Có thể thử nghiệm với các thuật toán RL khác nhau để so sánh hiệu suất.\n")
        
        self.logger.info(f"Đã tạo báo cáo huấn luyện tại {save_path}")
        return str(save_path)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê tóm tắt về quá trình huấn luyện.
        
        Returns:
            Dict chứa các thống kê
        """
        stats = {
            "experiment_name": self.experiment_name,
            "agent_name": self.agent_name,
            "env_name": self.env_name,
            "total_episodes": self.current_episode,
            "total_steps": self.total_steps,
            "training_duration": time.time() - self.start_time
        }
        
        # Thêm thống kê về phần thưởng
        rewards = self.metrics_history["episode_rewards"]
        if len(rewards) > 0:
            stats.update({
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "min_reward": float(np.min(rewards)),
                "max_reward": float(np.max(rewards)),
                "median_reward": float(np.median(rewards)),
                "last_reward": float(rewards[-1]),
                "mean_last_10": float(np.mean(rewards[-10:])) if len(rewards) >= 10 else None,
                "mean_last_100": float(np.mean(rewards[-100:])) if len(rewards) >= 100 else None
            })
        
        # Thêm thống kê về loss
        losses = self.metrics_history["losses"]
        if len(losses) > 0:
            stats.update({
                "mean_loss": float(np.mean(losses)),
                "last_loss": float(losses[-1]),
                "mean_last_10_loss": float(np.mean(losses[-10:])) if len(losses) >= 10 else None
            })
        
        # Thêm thống kê về win rate
        win_rates = [wr for wr in self.metrics_history["win_rates"] if wr is not None]
        if win_rates:
            stats.update({
                "mean_win_rate": float(np.mean(win_rates)),
                "last_win_rate": float(win_rates[-1]),
                "mean_last_10_win_rate": float(np.mean(win_rates[-10:])) if len(win_rates) >= 10 else None
            })
        
        return stats
    
    def compare_with_baseline(self, baseline_metrics_path: Union[str, Path]) -> Dict[str, Any]:
        """
        So sánh hiệu suất huấn luyện hiện tại với baseline.
        
        Args:
            baseline_metrics_path: Đường dẫn đến file số liệu baseline
            
        Returns:
            Dict chứa kết quả so sánh
        """
        try:
            # Tải dữ liệu baseline
            with open(baseline_metrics_path, 'r', encoding='utf-8') as f:
                baseline_data = json.load(f)
            
            # Kết quả so sánh
            comparison = {
                "current_experiment": self.experiment_name,
                "baseline_experiment": baseline_data.get("experiment_name", "Unknown"),
                "comparison_time": datetime.now().isoformat()
            }
            
            # So sánh các số liệu chính
            if "summary" in baseline_data and len(self.metrics_history["episode_rewards"]) > 0:
                baseline_summary = baseline_data["summary"]
                
                # Phần thưởng
                current_mean_reward = float(np.mean(self.metrics_history["episode_rewards"]))
                baseline_mean_reward = baseline_summary.get("mean_reward", 0)
                reward_improvement = current_mean_reward - baseline_mean_reward
                reward_improvement_percent = (reward_improvement / max(1e-10, abs(baseline_mean_reward))) * 100
                
                comparison["reward"] = {
                    "current_mean": current_mean_reward,
                    "baseline_mean": baseline_mean_reward,
                    "absolute_improvement": reward_improvement,
                    "percent_improvement": reward_improvement_percent
                }
                
                # Win rate
                current_win_rates = [wr for wr in self.metrics_history["win_rates"] if wr is not None]
                if current_win_rates and "mean_win_rate" in baseline_summary:
                    current_mean_win_rate = float(np.mean(current_win_rates))
                    baseline_mean_win_rate = baseline_summary.get("mean_win_rate", 0)
                    win_rate_improvement = current_mean_win_rate - baseline_mean_win_rate
                    win_rate_improvement_percent = (win_rate_improvement / max(1e-10, abs(baseline_mean_win_rate))) * 100
                    
                    comparison["win_rate"] = {
                        "current_mean": current_mean_win_rate,
                        "baseline_mean": baseline_mean_win_rate,
                        "absolute_improvement": win_rate_improvement,
                        "percent_improvement": win_rate_improvement_percent
                    }
                
                # Thời gian huấn luyện
                current_duration = time.time() - self.start_time
                baseline_duration = baseline_data.get("training_duration", 0)
                duration_change = current_duration - baseline_duration
                duration_change_percent = (duration_change / max(1e-10, abs(baseline_duration))) * 100
                
                comparison["training_duration"] = {
                    "current": current_duration,
                    "baseline": baseline_duration,
                    "absolute_change": duration_change,
                    "percent_change": duration_change_percent
                }
                
                # Kết luận tổng thể
                if reward_improvement > 0:
                    if current_duration < baseline_duration:
                        comparison["conclusion"] = "Mô hình hiện tại tốt hơn: Cải thiện hiệu suất và giảm thời gian huấn luyện."
                    else:
                        comparison["conclusion"] = "Mô hình hiện tại có hiệu suất tốt hơn nhưng tốn nhiều thời gian huấn luyện hơn."
                else:
                    if current_duration < baseline_duration:
                        comparison["conclusion"] = "Mô hình hiện tại nhanh hơn nhưng hiệu suất thấp hơn."
                    else:
                        comparison["conclusion"] = "Mô hình baseline vẫn tốt hơn: Hiệu suất cao hơn và thời gian huấn luyện ngắn hơn."
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Không thể so sánh với baseline: {str(e)}")
            return {
                "error": str(e),
                "current_experiment": self.experiment_name,
                "comparison_time": datetime.now().isoformat()
            }
    
    def reset(self) -> None:
        """
        Đặt lại trạng thái của tracker.
        """
        # Reset biến trạng thái
        self.current_episode = 0
        self.total_steps = 0
        self.start_time = time.time()
        self.episode_start_time = time.time()
        
        # Reset lịch sử số liệu
        for key in self.metrics_history:
            self.metrics_history[key] = []
        
        # Reset current metrics
        self.current_metrics = {
            "episode_reward": 0.0,
            "episode_length": 0,
            "losses": [],
            "win_count": 0,
            "loss_count": 0
        }
        
        self.logger.info("Đã reset TrainingMetricsTracker")
    
    def add_custom_metric(self, episode: int, metric_name: str, value: float) -> None:
        """
        Thêm số liệu tùy chỉnh.
        
        Args:
            episode: Số thứ tự episode
            metric_name: Tên số liệu
            value: Giá trị số liệu
        """
        # Thêm vào lịch sử nếu chưa có
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        # Đảm bảo đủ độ dài
        while len(self.metrics_history[metric_name]) < episode:
            self.metrics_history[metric_name].append(None)
        
        # Thêm hoặc cập nhật giá trị
        if len(self.metrics_history[metric_name]) == episode:
            self.metrics_history[metric_name].append(value)
        else:
            self.metrics_history[metric_name][episode - 1] = value
        
        # Ghi vào TensorBoard
        if self.use_tensorboard and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar(f"Custom/{metric_name}", value, step=episode)
                self.summary_writer.flush()
        
        self.logger.debug(f"Đã thêm số liệu tùy chỉnh: {metric_name}={value} cho episode {episode}")