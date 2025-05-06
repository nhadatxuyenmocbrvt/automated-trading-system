"""
Logger TensorBoard.
File này cung cấp các lớp và hàm để ghi log dữ liệu vào TensorBoard,
giúp trực quan hóa quá trình huấn luyện và theo dõi hiệu suất agent.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config, LOG_DIR

class TensorBoardLogger:
    """
    Lớp ghi log dữ liệu vào TensorBoard.
    Cung cấp các phương thức để ghi scalar, histogram, image, text, v.v.
    """
    
    def __init__(
        self,
        logdir: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        flush_secs: int = 120,
        max_queue: int = 10
    ):
        """
        Khởi tạo TensorBoard logger.
        
        Args:
            logdir: Thư mục lưu trữ log TensorBoard
            experiment_name: Tên thí nghiệm
            agent_name: Tên agent
            logger: Logger tùy chỉnh
            flush_secs: Thời gian tự động ghi dữ liệu vào đĩa (giây)
            max_queue: Số lượng event tối đa trong hàng đợi
        """
        # Thiết lập logger
        self.logger = logger or get_logger("tensorboard")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Xác định tên thí nghiệm
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.agent_name = agent_name or "agent"
        
        # Xác định thư mục log
        if logdir is None:
            if 'tensorboard_dir' in self.system_config:
                logdir = self.system_config['tensorboard_dir']
            else:
                logdir = LOG_DIR / "tensorboard"
        
        # Tạo đường dẫn đầy đủ
        self.log_dir = Path(logdir) / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo các writer cho từng thành phần
        try:
            self.writers = {
                "train": tf.summary.create_file_writer(
                    str(self.log_dir / "train"),
                    flush_millis=flush_secs * 1000,
                    max_queue=max_queue
                ),
                "val": tf.summary.create_file_writer(
                    str(self.log_dir / "val"),
                    flush_millis=flush_secs * 1000,
                    max_queue=max_queue
                ),
                "test": tf.summary.create_file_writer(
                    str(self.log_dir / "test"),
                    flush_millis=flush_secs * 1000,
                    max_queue=max_queue
                ),
                "custom": tf.summary.create_file_writer(
                    str(self.log_dir / "custom"),
                    flush_millis=flush_secs * 1000,
                    max_queue=max_queue
                ),
                "system": tf.summary.create_file_writer(
                    str(self.log_dir / "system"),
                    flush_millis=flush_secs * 1000,
                    max_queue=max_queue
                )
            }
            
            self.logger.info(f"Đã khởi tạo TensorBoardLogger tại {self.log_dir}")
        
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo TensorBoardLogger: {str(e)}")
            # Tạo writer trống nếu có lỗi
            self.writers = {}
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
        mode: str = "train"
    ) -> None:
        """
        Ghi giá trị scalar vào TensorBoard.
        
        Args:
            tag: Tên của scalar
            value: Giá trị scalar
            step: Bước huấn luyện
            mode: Chế độ log (train, val, test, custom)
        """
        if mode not in self.writers:
            self.logger.warning(f"Chế độ log không hợp lệ: {mode}")
            mode = "custom"
        
        try:
            with self.writers[mode].as_default():
                tf.summary.scalar(tag, value, step=step)
                self.writers[mode].flush()
        except Exception as e:
            self.logger.warning(f"Không thể ghi scalar {tag}: {str(e)}")
    
    def log_scalars(
        self,
        tag_value_dict: Dict[str, float],
        step: Optional[int] = None,
        mode: str = "train"
    ) -> None:
        """
        Ghi nhiều giá trị scalar vào TensorBoard.
        
        Args:
            tag_value_dict: Dict chứa cặp {tag: value}
            step: Bước huấn luyện
            mode: Chế độ log (train, val, test, custom)
        """
        if mode not in self.writers:
            self.logger.warning(f"Chế độ log không hợp lệ: {mode}")
            mode = "custom"
        
        try:
            with self.writers[mode].as_default():
                for tag, value in tag_value_dict.items():
                    tf.summary.scalar(tag, value, step=step)
                self.writers[mode].flush()
        except Exception as e:
            self.logger.warning(f"Không thể ghi scalars: {str(e)}")
    
    def log_histogram(
        self,
        tag: str,
        values: np.ndarray,
        step: Optional[int] = None,
        bins: Optional[int] = None,
        mode: str = "train"
    ) -> None:
        """
        Ghi histogram vào TensorBoard.
        
        Args:
            tag: Tên của histogram
            values: Dữ liệu cho histogram
            step: Bước huấn luyện
            bins: Số bins của histogram
            mode: Chế độ log (train, val, test, custom)
        """
        if mode not in self.writers:
            self.logger.warning(f"Chế độ log không hợp lệ: {mode}")
            mode = "custom"
        
        try:
            with self.writers[mode].as_default():
                tf.summary.histogram(tag, values, step=step, buckets=bins)
                self.writers[mode].flush()
        except Exception as e:
            self.logger.warning(f"Không thể ghi histogram {tag}: {str(e)}")
    
    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        step: Optional[int] = None,
        mode: str = "train",
        description: Optional[str] = None
    ) -> None:
        """
        Ghi hình ảnh vào TensorBoard.
        
        Args:
            tag: Tên của hình ảnh
            image: Dữ liệu hình ảnh (HxWxC hoặc BxHxWxC)
            step: Bước huấn luyện
            mode: Chế độ log (train, val, test, custom)
            description: Mô tả hình ảnh
        """
        if mode not in self.writers:
            self.logger.warning(f"Chế độ log không hợp lệ: {mode}")
            mode = "custom"
        
        try:
            # Đảm bảo shape phù hợp (BxHxWxC)
            if len(image.shape) == 3:
                image = image[np.newaxis, :]
            
            with self.writers[mode].as_default():
                tf.summary.image(tag, image, step=step, description=description)
                self.writers[mode].flush()
        except Exception as e:
            self.logger.warning(f"Không thể ghi hình ảnh {tag}: {str(e)}")
    
    def log_figure(
        self,
        tag: str,
        figure: Figure,
        step: Optional[int] = None,
        mode: str = "train",
        close_figure: bool = True
    ) -> None:
        """
        Ghi matplotlib figure vào TensorBoard.
        
        Args:
            tag: Tên của figure
            figure: Matplotlib figure
            step: Bước huấn luyện
            mode: Chế độ log (train, val, test, custom)
            close_figure: Tự động đóng figure sau khi ghi
        """
        if mode not in self.writers:
            self.logger.warning(f"Chế độ log không hợp lệ: {mode}")
            mode = "custom"
        
        try:
            # Chuyển đổi figure thành hình ảnh
            buf = io.BytesIO()
            figure.savefig(buf, format='png')
            buf.seek(0)
            
            # Đọc hình ảnh từ buffer
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)  # Thêm batch dimension
            
            # Ghi hình ảnh
            with self.writers[mode].as_default():
                tf.summary.image(tag, image, step=step)
                self.writers[mode].flush()
            
            # Đóng figure nếu cần
            if close_figure:
                plt.close(figure)
                
        except Exception as e:
            self.logger.warning(f"Không thể ghi figure {tag}: {str(e)}")
    
    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None,
        mode: str = "train"
    ) -> None:
        """
        Ghi text vào TensorBoard.
        
        Args:
            tag: Tên của text
            text: Nội dung text
            step: Bước huấn luyện
            mode: Chế độ log (train, val, test, custom)
        """
        if mode not in self.writers:
            self.logger.warning(f"Chế độ log không hợp lệ: {mode}")
            mode = "custom"
        
        try:
            with self.writers[mode].as_default():
                tf.summary.text(tag, text, step=step)
                self.writers[mode].flush()
        except Exception as e:
            self.logger.warning(f"Không thể ghi text {tag}: {str(e)}")
    
    def log_embedding(
        self,
        tag: str,
        embeddings: np.ndarray,
        metadata: Optional[List[str]] = None,
        images: Optional[np.ndarray] = None,
        step: Optional[int] = None,
        mode: str = "train"
    ) -> None:
        """
        Ghi embedding vào TensorBoard.
        
        Args:
            tag: Tên của embedding
            embeddings: Ma trận embedding shape (N, D)
            metadata: Danh sách metadata cho từng embedding
            images: Hình ảnh tương ứng với từng embedding
            step: Bước huấn luyện
            mode: Chế độ log (train, val, test, custom)
        """
        if mode not in self.writers:
            self.logger.warning(f"Chế độ log không hợp lệ: {mode}")
            mode = "custom"
        
        try:
            # TensorFlow không có hàm trực tiếp cho embeddings như PyTorch
            # Thay vào đó, ta lưu file để projector đọc
            from tensorboard.plugins import projector
            import tensorflow as tf
            
            # Tạo thư mục log
            log_subdir = self.log_dir / mode / f"embedding_{tag}_{step or 0}"
            log_subdir.mkdir(parents=True, exist_ok=True)
            
            # Tạo và lưu tensor
            tensor_path = log_subdir / "embeddings.ckpt"
            tensor = tf.Variable(embeddings)
            checkpoint = tf.train.Checkpoint(embedding=tensor)
            checkpoint.save(str(tensor_path))
            
            # Cấu hình projector
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
            embedding.tensor_path = str(tensor_path) + "-1.data-00000-of-00001"
            
            # Lưu metadata nếu có
            if metadata is not None:
                metadata_path = log_subdir / "metadata.tsv"
                with open(metadata_path, 'w') as f:
                    for meta in metadata:
                        f.write(f"{meta}\n")
                embedding.metadata_path = str(metadata_path)
            
            # Lưu hình ảnh nếu có
            if images is not None:
                # Cần lưu hình ảnh theo định dạng đặc biệt cho projector
                self.logger.warning("Chức năng lưu hình ảnh cho embeddings chưa được hỗ trợ đầy đủ")
            
            # Lưu cấu hình projector
            projector.visualize_embeddings(str(log_subdir), config)
            
            self.logger.info(f"Đã lưu embedding {tag} tại {log_subdir}")
            
        except Exception as e:
            self.logger.warning(f"Không thể ghi embedding {tag}: {str(e)}")
    
    def log_graph(
        self,
        model: tf.keras.Model,
        input_shape: Optional[Tuple[int, ...]] = None
    ) -> None:
        """
        Ghi mô hình graph vào TensorBoard.
        
        Args:
            model: Mô hình Keras
            input_shape: Shape của input
        """
        try:
            if input_shape is not None:
                # Tạo input giả để xác định các shape bên trong mạng
                inputs = tf.random.normal(input_shape)
                # Gọi model trên input để build
                _ = model(inputs)
            
            # Sử dụng writer custom
            with self.writers["custom"].as_default():
                # Ghi graph
                tf.summary.trace_on(graph=True, profiler=False)
                # Running the model again will trace the graph
                if input_shape is not None:
                    _ = model(inputs)
                tf.summary.trace_export(name="model_graph", step=0)
                self.writers["custom"].flush()
            
            self.logger.info(f"Đã ghi graph mô hình {model.name}")
            
        except Exception as e:
            self.logger.warning(f"Không thể ghi graph mô hình: {str(e)}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Ghi hyperparameters vào TensorBoard.
        
        Args:
            hparams: Dict chứa hyperparameters
            step: Bước huấn luyện
        """
        try:
            # Chuyển đổi tất cả các giá trị sang định dạng phù hợp
            hp_dict = {}
            for name, value in hparams.items():
                if isinstance(value, (int, float, str, bool)):
                    hp_dict[name] = value
                else:
                    # Chuyển đổi sang string nếu không phải các kiểu cơ bản
                    hp_dict[name] = str(value)
            
            # Sử dụng writer custom
            with self.writers["custom"].as_default():
                # Ghi hyperparameters dưới dạng text
                hp_text = "\n".join([f"{k}: {v}" for k, v in hp_dict.items()])
                tf.summary.text("hyperparameters", hp_text, step=step)
                
                # Ghi từng hyperparameter dưới dạng scalar nếu có thể
                for name, value in hp_dict.items():
                    if isinstance(value, (int, float)):
                        tf.summary.scalar(f"hparams/{name}", value, step=step)
                
                self.writers["custom"].flush()
            
            self.logger.info(f"Đã ghi {len(hp_dict)} hyperparameters")
            
        except Exception as e:
            self.logger.warning(f"Không thể ghi hyperparameters: {str(e)}")
    
    def log_system_metrics(
        self,
        cpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None,
        gpu_usage: Optional[float] = None,
        gpu_memory_usage: Optional[float] = None,
        disk_usage: Optional[float] = None,
        step: Optional[int] = None
    ) -> None:
        """
        Ghi các metrics hệ thống vào TensorBoard.
        
        Args:
            cpu_usage: Phần trăm sử dụng CPU
            memory_usage: Phần trăm sử dụng RAM
            gpu_usage: Phần trăm sử dụng GPU
            gpu_memory_usage: Phần trăm sử dụng bộ nhớ GPU
            disk_usage: Phần trăm sử dụng ổ đĩa
            step: Bước huấn luyện
        """
        metrics = {}
        
        if cpu_usage is not None:
            metrics["system/cpu_usage"] = cpu_usage
        
        if memory_usage is not None:
            metrics["system/memory_usage"] = memory_usage
        
        if gpu_usage is not None:
            metrics["system/gpu_usage"] = gpu_usage
        
        if gpu_memory_usage is not None:
            metrics["system/gpu_memory_usage"] = gpu_memory_usage
        
        if disk_usage is not None:
            metrics["system/disk_usage"] = disk_usage
        
        try:
            with self.writers["system"].as_default():
                for tag, value in metrics.items():
                    tf.summary.scalar(tag, value, step=step)
                self.writers["system"].flush()
        except Exception as e:
            self.logger.warning(f"Không thể ghi system metrics: {str(e)}")
    
    def log_agent_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
        mode: str = "train"
    ) -> None:
        """
        Ghi các metrics của agent vào TensorBoard.
        
        Args:
            metrics: Dict chứa các metrics
            step: Bước huấn luyện
            prefix: Tiền tố cho tên metrics
            mode: Chế độ log (train, val, test, custom)
        """
        if mode not in self.writers:
            self.logger.warning(f"Chế độ log không hợp lệ: {mode}")
            mode = "custom"
        
        try:
            with self.writers[mode].as_default():
                for name, value in metrics.items():
                    # Thêm tiền tố nếu có
                    tag = f"{prefix}/{name}" if prefix else name
                    tf.summary.scalar(tag, value, step=step)
                self.writers[mode].flush()
        except Exception as e:
            self.logger.warning(f"Không thể ghi agent metrics: {str(e)}")
    
    def log_episode_metrics(
        self, 
        episode: int,
        rewards: List[float],
        losses: Optional[List[float]] = None,
        epsilon: Optional[float] = None,
        length: Optional[int] = None,
        win_rate: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None,
        mode: str = "train"
    ) -> None:
        """
        Ghi các metrics của episode vào TensorBoard.
        
        Args:
            episode: Số thứ tự episode
            rewards: Danh sách phần thưởng trong episode
            losses: Danh sách loss trong episode
            epsilon: Giá trị epsilon
            length: Độ dài episode
            win_rate: Tỷ lệ thắng
            additional_metrics: Các metrics bổ sung
            mode: Chế độ log (train, val, test, custom)
        """
        if mode not in self.writers:
            self.logger.warning(f"Chế độ log không hợp lệ: {mode}")
            mode = "custom"
            
        # Tính toán các metrics
        total_reward = sum(rewards)
        mean_reward = np.mean(rewards) if rewards else 0
        
        try:
            with self.writers[mode].as_default():
                # Ghi các metrics cơ bản
                tf.summary.scalar("episode/total_reward", total_reward, step=episode)
                tf.summary.scalar("episode/mean_reward", mean_reward, step=episode)
                
                if losses:
                    mean_loss = np.mean(losses)
                    tf.summary.scalar("episode/mean_loss", mean_loss, step=episode)
                
                if epsilon is not None:
                    tf.summary.scalar("episode/epsilon", epsilon, step=episode)
                
                if length is not None:
                    tf.summary.scalar("episode/length", length, step=episode)
                
                if win_rate is not None:
                    tf.summary.scalar("episode/win_rate", win_rate, step=episode)
                
                # Ghi các metrics bổ sung
                if additional_metrics:
                    for name, value in additional_metrics.items():
                        tf.summary.scalar(f"episode/{name}", value, step=episode)
                
                # Ghi histogram phần thưởng
                if rewards:
                    tf.summary.histogram("episode/rewards_distribution", rewards, step=episode)
                
                # Ghi histogram loss
                if losses:
                    tf.summary.histogram("episode/losses_distribution", losses, step=episode)
                
                self.writers[mode].flush()
                
        except Exception as e:
            self.logger.warning(f"Không thể ghi episode metrics: {str(e)}")
    
    def log_model_weights(
        self,
        model: tf.keras.Model,
        step: Optional[int] = None,
        mode: str = "train"
    ) -> None:
        """
        Ghi các trọng số của mô hình vào TensorBoard.
        
        Args:
            model: Mô hình Keras
            step: Bước huấn luyện
            mode: Chế độ log (train, val, test, custom)
        """
        if mode not in self.writers:
            self.logger.warning(f"Chế độ log không hợp lệ: {mode}")
            mode = "custom"
        
        try:
            with self.writers[mode].as_default():
                # Ghi histogram cho mỗi layer
                for layer in model.layers:
                    for weight in layer.weights:
                        weight_name = weight.name.replace(':', '_')
                        tf.summary.histogram(f"weights/{layer.name}/{weight_name}", weight, step=step)
                
                self.writers[mode].flush()
                
        except Exception as e:
            self.logger.warning(f"Không thể ghi model weights: {str(e)}")
    
    def log_model_gradients(
        self,
        gradients: List[tf.Tensor],
        weights: List[tf.Tensor],
        step: Optional[int] = None,
        mode: str = "train"
    ) -> None:
        """
        Ghi các gradient của mô hình vào TensorBoard.
        
        Args:
            gradients: Danh sách gradient
            weights: Danh sách trọng số tương ứng
            step: Bước huấn luyện
            mode: Chế độ log (train, val, test, custom)
        """
        if mode not in self.writers:
            self.logger.warning(f"Chế độ log không hợp lệ: {mode}")
            mode = "custom"
        
        try:
            with self.writers[mode].as_default():
                # Ghi histogram và norm cho mỗi gradient
                for i, (grad, var) in enumerate(zip(gradients, weights)):
                    if grad is not None:
                        var_name = var.name.replace(':', '_')
                        
                        # Histogram gradient
                        tf.summary.histogram(f"gradients/{var_name}", grad, step=step)
                        
                        # Norm gradient
                        grad_norm = tf.norm(grad)
                        tf.summary.scalar(f"gradient_norm/{var_name}", grad_norm, step=step)
                
                self.writers[mode].flush()
                
        except Exception as e:
            self.logger.warning(f"Không thể ghi model gradients: {str(e)}")
    
    def flush(self) -> None:
        """
        Ghi tất cả dữ liệu đang trong bộ đệm ra đĩa.
        """
        for writer in self.writers.values():
            try:
                writer.flush()
            except Exception as e:
                self.logger.warning(f"Không thể flush writer: {str(e)}")
    
    def close(self) -> None:
        """
        Đóng tất cả các writer và giải phóng tài nguyên.
        """
        for writer in self.writers.values():
            try:
                writer.close()
            except Exception as e:
                self.logger.warning(f"Không thể đóng writer: {str(e)}")
        
        self.logger.info("Đã đóng tất cả TensorBoard writers")

# Singleton instance
_tb_logger_instance = None

def get_tb_logger(
    logdir: Optional[Union[str, Path]] = None,
    experiment_name: Optional[str] = None,
    agent_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    flush_secs: int = 120,
    max_queue: int = 10,
    force_new: bool = False
) -> TensorBoardLogger:
    """
    Hàm helper để lấy instance TensorBoardLogger.
    
    Args:
        logdir: Thư mục lưu trữ log TensorBoard
        experiment_name: Tên thí nghiệm
        agent_name: Tên agent
        logger: Logger tùy chỉnh
        flush_secs: Thời gian tự động ghi dữ liệu vào đĩa (giây)
        max_queue: Số lượng event tối đa trong hàng đợi
        force_new: Tạo instance mới ngay cả khi đã tồn tại
        
    Returns:
        Instance TensorBoardLogger
    """
    global _tb_logger_instance
    
    if _tb_logger_instance is None or force_new:
        _tb_logger_instance = TensorBoardLogger(
            logdir=logdir,
            experiment_name=experiment_name,
            agent_name=agent_name,
            logger=logger,
            flush_secs=flush_secs,
            max_queue=max_queue
        )
    
    return _tb_logger_instance

def log_scalar(tag: str, value: float, step: Optional[int] = None, mode: str = "train") -> None:
    """
    Hàm helper để ghi giá trị scalar vào TensorBoard.
    
    Args:
        tag: Tên của scalar
        value: Giá trị scalar
        step: Bước huấn luyện
        mode: Chế độ log (train, val, test, custom)
    """
    logger = get_tb_logger()
    logger.log_scalar(tag, value, step, mode)

def log_scalars(tag_value_dict: Dict[str, float], step: Optional[int] = None, mode: str = "train") -> None:
    """
    Hàm helper để ghi nhiều giá trị scalar vào TensorBoard.
    
    Args:
        tag_value_dict: Dict chứa cặp {tag: value}
        step: Bước huấn luyện
        mode: Chế độ log (train, val, test, custom)
    """
    logger = get_tb_logger()
    logger.log_scalars(tag_value_dict, step, mode)

def log_episode_metrics(
    episode: int,
    rewards: List[float],
    losses: Optional[List[float]] = None,
    epsilon: Optional[float] = None,
    length: Optional[int] = None,
    win_rate: Optional[float] = None,
    additional_metrics: Optional[Dict[str, float]] = None,
    mode: str = "train"
) -> None:
    """
    Hàm helper để ghi các metrics của episode vào TensorBoard.
    
    Args:
        episode: Số thứ tự episode
        rewards: Danh sách phần thưởng trong episode
        losses: Danh sách loss trong episode
        epsilon: Giá trị epsilon
        length: Độ dài episode
        win_rate: Tỷ lệ thắng
        additional_metrics: Các metrics bổ sung
        mode: Chế độ log (train, val, test, custom)
    """
    logger = get_tb_logger()
    logger.log_episode_metrics(
        episode, rewards, losses, epsilon, length, win_rate, additional_metrics, mode
    )

# Main execution
if __name__ == "__main__":
    # Một ví dụ nhỏ về cách sử dụng
    logger = get_tb_logger(experiment_name="example_experiment")
    
    # Log một số scalar
    for i in range(100):
        logger.log_scalar("test/value", np.sin(i / 10.0), step=i)
        logger.log_scalar("test/value_squared", np.sin(i / 10.0) ** 2, step=i)
    
    # Log histogram
    logger.log_histogram("test/histogram", np.random.randn(1000), step=0)
    
    # Log hình ảnh
    img = np.random.rand(32, 32, 3)
    logger.log_image("test/random_image", img, step=0)
    
    # Đóng logger
    logger.close()
    
    print("Đã ghi log xong. Chạy 'tensorboard --logdir=logs/tensorboard' để xem kết quả.")