"""
Đo lường tùy chỉnh cho TensorBoard.
File này cung cấp các lớp và hàm để tạo và ghi lại các số liệu tùy chỉnh
cho TensorBoard, hỗ trợ theo dõi quá trình huấn luyện và giao dịch một cách
trực quan và chi tiết hơn.
"""

import os
import time
import logging
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.ops import summary_ops_v2
# Import projector cho visualizing embeddings
from tensorboard.plugins import projector
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from config.constants import Timeframe, BacktestMetric, AgentType

class TensorBoardCustomMetric:
    """
    Lớp cơ sở cho các số liệu tùy chỉnh cho TensorBoard.
    Cung cấp các phương thức chung để khởi tạo, cập nhật và hiển thị
    các số liệu tùy chỉnh trên TensorBoard.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        log_dir: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None,
        dtype: tf.DType = tf.float32,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo số liệu tùy chỉnh.
        
        Args:
            name: Tên của số liệu
            description: Mô tả về số liệu
            log_dir: Thư mục lưu log TensorBoard
            experiment_name: Tên thí nghiệm
            dtype: Kiểu dữ liệu của số liệu
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("tensorboard_metrics")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập các thuộc tính
        self.name = name
        self.description = description
        self.dtype = dtype
        
        # Thiết lập thư mục log
        if log_dir is None:
            logs_dir = Path(self.system_config.get("log_dir", "./logs"))
            log_dir = logs_dir / "tensorboard"
        else:
            log_dir = Path(log_dir)
        
        # Thêm tên thí nghiệm nếu có
        if experiment_name is not None:
            log_dir = log_dir / experiment_name
        
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo writer TensorBoard
        self.summary_writer = tf.summary.create_file_writer(str(self.log_dir))
        
        # Biến theo dõi giá trị
        self.values = []
        self.steps = []
        self.last_step = 0
        self.last_value = None
        
        self.logger.info(f"Khởi tạo số liệu tùy chỉnh '{name}' cho TensorBoard tại {self.log_dir}")
    
    def update(self, value: Any, step: Optional[int] = None) -> None:
        """
        Cập nhật giá trị cho số liệu.
        
        Args:
            value: Giá trị mới
            step: Bước hiện tại (None để tự động tăng)
        """
        # Tự động tăng step nếu không được cung cấp
        if step is None:
            step = self.last_step + 1
        
        # Lưu giá trị và step
        self.values.append(value)
        self.steps.append(step)
        self.last_step = step
        self.last_value = value
        
        # Ghi vào TensorBoard
        self._write_to_tensorboard(value, step)
    
    def _write_to_tensorboard(self, value: Any, step: int) -> None:
        """
        Ghi giá trị vào TensorBoard.
        Được ghi đè bởi lớp con tùy thuộc vào loại số liệu.
        
        Args:
            value: Giá trị cần ghi
            step: Bước hiện tại
        """
        with self.summary_writer.as_default():
            tf.summary.scalar(self.name, value, step=step)
            self.summary_writer.flush()
    
    def get_latest(self) -> Tuple[Any, int]:
        """
        Lấy giá trị mới nhất của số liệu.
        
        Returns:
            Tuple (giá trị, step)
        """
        if not self.values:
            return None, 0
        return self.last_value, self.last_step
    
    def get_history(self) -> Tuple[List[Any], List[int]]:
        """
        Lấy toàn bộ lịch sử giá trị và step.
        
        Returns:
            Tuple (danh sách giá trị, danh sách step)
        """
        return self.values, self.steps
    
    def get_average(self, window_size: Optional[int] = None) -> float:
        """
        Tính giá trị trung bình của số liệu.
        
        Args:
            window_size: Số lượng giá trị gần nhất để tính trung bình
            
        Returns:
            Giá trị trung bình
        """
        if not self.values:
            return 0.0
        
        if window_size is not None and window_size > 0:
            values_to_avg = self.values[-min(window_size, len(self.values)):]
        else:
            values_to_avg = self.values
        
        return np.mean(values_to_avg)
    
    def reset(self) -> None:
        """
        Đặt lại số liệu về trạng thái ban đầu.
        """
        self.values = []
        self.steps = []
        self.last_step = 0
        self.last_value = None
        self.logger.debug(f"Đã đặt lại số liệu '{self.name}'")


class ScalarMetric(TensorBoardCustomMetric):
    """
    Số liệu vô hướng đơn giản cho TensorBoard.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        log_dir: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None,
        dtype: tf.DType = tf.float32,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(name, description, log_dir, experiment_name, dtype, logger)
    
    def _write_to_tensorboard(self, value: float, step: int) -> None:
        """
        Ghi giá trị vô hướng vào TensorBoard.
        
        Args:
            value: Giá trị cần ghi
            step: Bước hiện tại
        """
        with self.summary_writer.as_default():
            tf.summary.scalar(self.name, value, step=step)
            self.summary_writer.flush()


class HistogramMetric(TensorBoardCustomMetric):
    """
    Số liệu histogram cho TensorBoard.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        log_dir: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None,
        dtype: tf.DType = tf.float32,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(name, description, log_dir, experiment_name, dtype, logger)
    
    def _write_to_tensorboard(self, value: Union[List[float], np.ndarray], step: int) -> None:
        """
        Ghi histogram vào TensorBoard.
        
        Args:
            value: Dữ liệu cho histogram
            step: Bước hiện tại
        """
        with self.summary_writer.as_default():
            tf.summary.histogram(self.name, value, step=step)
            self.summary_writer.flush()


class ImageMetric(TensorBoardCustomMetric):
    """
    Số liệu hình ảnh cho TensorBoard.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        log_dir: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None,
        max_outputs: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(name, description, log_dir, experiment_name, tf.uint8, logger)
        self.max_outputs = max_outputs
    
    def _write_to_tensorboard(self, value: Union[np.ndarray, List[np.ndarray]], step: int) -> None:
        """
        Ghi hình ảnh vào TensorBoard.
        
        Args:
            value: Dữ liệu hình ảnh
            step: Bước hiện tại
        """
        with self.summary_writer.as_default():
            # Kiểm tra nếu value là một hình ảnh duy nhất hay danh sách hình ảnh
            if isinstance(value, np.ndarray) and value.ndim in [3, 4]:
                if value.ndim == 3:  # Một hình ảnh duy nhất
                    value = value[np.newaxis, ...]
                
                # Giới hạn số lượng hình ảnh hiển thị
                value = value[:self.max_outputs]
                
                tf.summary.image(self.name, value, step=step, max_outputs=self.max_outputs)
                self.summary_writer.flush()
            else:
                self.logger.warning(f"Định dạng hình ảnh không hợp lệ cho '{self.name}'")


class TextMetric(TensorBoardCustomMetric):
    """
    Số liệu văn bản cho TensorBoard.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        log_dir: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(name, description, log_dir, experiment_name, tf.string, logger)
    
    def _write_to_tensorboard(self, value: str, step: int) -> None:
        """
        Ghi văn bản vào TensorBoard.
        
        Args:
            value: Văn bản cần ghi
            step: Bước hiện tại
        """
        with self.summary_writer.as_default():
            tf.summary.text(self.name, value, step=step)
            self.summary_writer.flush()


class ConfusionMatrixMetric(TensorBoardCustomMetric):
    """
    Ma trận nhầm lẫn cho TensorBoard.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        class_names: List[str],
        log_dir: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(name, description, log_dir, experiment_name, tf.int32, logger)
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def _write_to_tensorboard(self, value: np.ndarray, step: int) -> None:
        """
        Ghi ma trận nhầm lẫn vào TensorBoard.
        
        Args:
            value: Ma trận nhầm lẫn
            step: Bước hiện tại
        """
        figure = self._plot_confusion_matrix(value)
        
        with self.summary_writer.as_default():
            tf.summary.image(self.name, self._plot_to_image(figure), step=step)
            self.summary_writer.flush()
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """
        Tạo biểu đồ ma trận nhầm lẫn.
        
        Args:
            cm: Ma trận nhầm lẫn
            
        Returns:
            Đối tượng figure matplotlib
        """
        import matplotlib.pyplot as plt
        
        figure = plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"{self.name} - Confusion Matrix")
        plt.colorbar()
        
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)
        
        # Hiển thị giá trị trong ma trận
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        return figure
    
    def _plot_to_image(self, figure):
        """
        Chuyển đổi biểu đồ matplotlib thành hình ảnh PNG.
        
        Args:
            figure: Đối tượng figure matplotlib
            
        Returns:
            Hình ảnh dưới dạng tensor
        """
        import io
        import matplotlib.pyplot as plt
        
        # Lưu biểu đồ vào bộ đệm BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        
        # Đọc PNG thành tensor
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        
        return image


class EmbeddingMetric(TensorBoardCustomMetric):
    """
    Số liệu embedding cho TensorBoard.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        log_dir: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None,
        metadata: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(name, description, log_dir, experiment_name, tf.float32, logger)
        self.metadata = metadata
    
    def _write_to_tensorboard(self, value: np.ndarray, step: int) -> None:
        """
        Ghi embedding vào TensorBoard.
        
        Args:
            value: Dữ liệu embedding
            step: Bước hiện tại
        """
        import tensorflow as tf
        
        # Thư mục lưu thông tin embedding
        log_dir = self.log_dir / f"step_{step}" / self.name
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo và lưu metadata nếu có
        metadata_path = None
        if self.metadata is not None:
            metadata_path = log_dir / "metadata.tsv"
            with open(metadata_path, 'w') as f:
                for meta in self.metadata:
                    f.write(f"{meta}\n")
        
        # Lưu embedding
        checkpoint = tf.train.Checkpoint(embedding=tf.Variable(value))
        checkpoint.save(str(log_dir / "embedding.ckpt"))
        
        # Cấu hình projector
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.tensor_path = str(log_dir / "embedding.ckpt.index")
        
        if metadata_path is not None:
            embedding.metadata_path = str(metadata_path)
        
        projector.visualize_embeddings(str(log_dir), config)


class HyperParameterMetric(TensorBoardCustomMetric):
    """
    Số liệu siêu tham số cho TensorBoard.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        log_dir: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(name, description, log_dir, experiment_name, tf.string, logger)
    
    def _write_to_tensorboard(self, value: Dict[str, Any], step: int) -> None:
        """
        Ghi siêu tham số vào TensorBoard.
        
        Args:
            value: Dict chứa các siêu tham số
            step: Bước hiện tại
        """
        # Ghi các siêu tham số riêng lẻ
        with self.summary_writer.as_default():
            for param_name, param_value in value.items():
                param_tag = f"{self.name}/{param_name}"
                
                # Xử lý tùy theo loại dữ liệu
                if isinstance(param_value, (int, float)):
                    tf.summary.scalar(param_tag, param_value, step=step)
                elif isinstance(param_value, str):
                    tf.summary.text(param_tag, param_value, step=step)
                elif isinstance(param_value, (list, np.ndarray)) and all(isinstance(x, (int, float)) for x in param_value):
                    tf.summary.histogram(param_tag, param_value, step=step)
                else:
                    # Chuyển đổi sang text nếu là dữ liệu phức tạp
                    tf.summary.text(param_tag, str(param_value), step=step)
            
            # Ghi toàn bộ siêu tham số dưới dạng JSON
            tf.summary.text(f"{self.name}/all_params", json.dumps(value, indent=2, default=str), step=step)
            
            self.summary_writer.flush()


class MetricsManager:
    """
    Lớp quản lý các số liệu tùy chỉnh TensorBoard.
    Cung cấp các phương thức để tạo, truy cập và cập nhật các số liệu
    cho nhiều thí nghiệm.
    """
    
    def __init__(
        self,
        root_log_dir: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo quản lý số liệu.
        
        Args:
            root_log_dir: Thư mục gốc cho log TensorBoard
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("tensorboard_manager")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập thư mục gốc
        if root_log_dir is None:
            logs_dir = Path(self.system_config.get("log_dir", "./logs"))
            root_log_dir = logs_dir / "tensorboard"
        
        self.root_log_dir = Path(root_log_dir)
        self.root_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Dict chứa các số liệu theo experiment
        self.metrics = defaultdict(dict)
        
        self.logger.info(f"Khởi tạo TensorBoard MetricsManager tại {self.root_log_dir}")
    
    def create_scalar_metric(
        self,
        name: str,
        description: str,
        experiment_name: str,
        dtype: tf.DType = tf.float32
    ) -> ScalarMetric:
        """
        Tạo số liệu vô hướng mới.
        
        Args:
            name: Tên của số liệu
            description: Mô tả về số liệu
            experiment_name: Tên thí nghiệm
            dtype: Kiểu dữ liệu của số liệu
            
        Returns:
            Đối tượng ScalarMetric
        """
        # Tạo thư mục log cho experiment
        log_dir = self.root_log_dir / experiment_name
        
        # Tạo số liệu mới
        metric = ScalarMetric(
            name=name,
            description=description,
            log_dir=log_dir,
            experiment_name=None,  # Đã bao gồm trong log_dir
            dtype=dtype,
            logger=self.logger
        )
        
        # Lưu vào dict
        self.metrics[experiment_name][name] = metric
        
        return metric
    
    def create_histogram_metric(
        self,
        name: str,
        description: str,
        experiment_name: str,
        dtype: tf.DType = tf.float32
    ) -> HistogramMetric:
        """
        Tạo số liệu histogram mới.
        
        Args:
            name: Tên của số liệu
            description: Mô tả về số liệu
            experiment_name: Tên thí nghiệm
            dtype: Kiểu dữ liệu của số liệu
            
        Returns:
            Đối tượng HistogramMetric
        """
        # Tạo thư mục log cho experiment
        log_dir = self.root_log_dir / experiment_name
        
        # Tạo số liệu mới
        metric = HistogramMetric(
            name=name,
            description=description,
            log_dir=log_dir,
            experiment_name=None,  # Đã bao gồm trong log_dir
            dtype=dtype,
            logger=self.logger
        )
        
        # Lưu vào dict
        self.metrics[experiment_name][name] = metric
        
        return metric
    
    def create_image_metric(
        self,
        name: str,
        description: str,
        experiment_name: str,
        max_outputs: int = 3
    ) -> ImageMetric:
        """
        Tạo số liệu hình ảnh mới.
        
        Args:
            name: Tên của số liệu
            description: Mô tả về số liệu
            experiment_name: Tên thí nghiệm
            max_outputs: Số lượng hình ảnh tối đa hiển thị
            
        Returns:
            Đối tượng ImageMetric
        """
        # Tạo thư mục log cho experiment
        log_dir = self.root_log_dir / experiment_name
        
        # Tạo số liệu mới
        metric = ImageMetric(
            name=name,
            description=description,
            log_dir=log_dir,
            experiment_name=None,  # Đã bao gồm trong log_dir
            max_outputs=max_outputs,
            logger=self.logger
        )
        
        # Lưu vào dict
        self.metrics[experiment_name][name] = metric
        
        return metric
    
    def create_text_metric(
        self,
        name: str,
        description: str,
        experiment_name: str
    ) -> TextMetric:
        """
        Tạo số liệu văn bản mới.
        
        Args:
            name: Tên của số liệu
            description: Mô tả về số liệu
            experiment_name: Tên thí nghiệm
            
        Returns:
            Đối tượng TextMetric
        """
        # Tạo thư mục log cho experiment
        log_dir = self.root_log_dir / experiment_name
        
        # Tạo số liệu mới
        metric = TextMetric(
            name=name,
            description=description,
            log_dir=log_dir,
            experiment_name=None,  # Đã bao gồm trong log_dir
            logger=self.logger
        )
        
        # Lưu vào dict
        self.metrics[experiment_name][name] = metric
        
        return metric
    
    def create_confusion_matrix_metric(
        self,
        name: str,
        description: str,
        experiment_name: str,
        class_names: List[str]
    ) -> ConfusionMatrixMetric:
        """
        Tạo số liệu ma trận nhầm lẫn mới.
        
        Args:
            name: Tên của số liệu
            description: Mô tả về số liệu
            experiment_name: Tên thí nghiệm
            class_names: Danh sách tên các lớp
            
        Returns:
            Đối tượng ConfusionMatrixMetric
        """
        # Tạo thư mục log cho experiment
        log_dir = self.root_log_dir / experiment_name
        
        # Tạo số liệu mới
        metric = ConfusionMatrixMetric(
            name=name,
            description=description,
            class_names=class_names,
            log_dir=log_dir,
            experiment_name=None,  # Đã bao gồm trong log_dir
            logger=self.logger
        )
        
        # Lưu vào dict
        self.metrics[experiment_name][name] = metric
        
        return metric
    
    def create_hyperparameter_metric(
        self,
        name: str,
        description: str,
        experiment_name: str
    ) -> HyperParameterMetric:
        """
        Tạo số liệu siêu tham số mới.
        
        Args:
            name: Tên của số liệu
            description: Mô tả về số liệu
            experiment_name: Tên thí nghiệm
            
        Returns:
            Đối tượng HyperParameterMetric
        """
        # Tạo thư mục log cho experiment
        log_dir = self.root_log_dir / experiment_name
        
        # Tạo số liệu mới
        metric = HyperParameterMetric(
            name=name,
            description=description,
            log_dir=log_dir,
            experiment_name=None,  # Đã bao gồm trong log_dir
            logger=self.logger
        )
        
        # Lưu vào dict
        self.metrics[experiment_name][name] = metric
        
        return metric
    
    def get_metric(self, experiment_name: str, metric_name: str) -> Optional[TensorBoardCustomMetric]:
        """
        Lấy số liệu theo tên.
        
        Args:
            experiment_name: Tên thí nghiệm
            metric_name: Tên số liệu
            
        Returns:
            Đối tượng số liệu hoặc None nếu không tìm thấy
        """
        if experiment_name in self.metrics and metric_name in self.metrics[experiment_name]:
            return self.metrics[experiment_name][metric_name]
        return None
    
    def get_all_metrics(self, experiment_name: str) -> Dict[str, TensorBoardCustomMetric]:
        """
        Lấy tất cả số liệu của một thí nghiệm.
        
        Args:
            experiment_name: Tên thí nghiệm
            
        Returns:
            Dict chứa các số liệu
        """
        return self.metrics.get(experiment_name, {})
    
    def update_metric(
        self,
        experiment_name: str,
        metric_name: str,
        value: Any,
        step: Optional[int] = None
    ) -> bool:
        """
        Cập nhật giá trị cho một số liệu.
        
        Args:
            experiment_name: Tên thí nghiệm
            metric_name: Tên số liệu
            value: Giá trị mới
            step: Bước hiện tại
            
        Returns:
            True nếu cập nhật thành công, False nếu không
        """
        metric = self.get_metric(experiment_name, metric_name)
        if metric is not None:
            metric.update(value, step)
            return True
        
        self.logger.warning(f"Không tìm thấy số liệu '{metric_name}' trong thí nghiệm '{experiment_name}'")
        return False
    
    def batch_update(
        self,
        experiment_name: str,
        metrics_values: Dict[str, Any],
        step: Optional[int] = None
    ) -> Dict[str, bool]:
        """
        Cập nhật đồng thời nhiều số liệu.
        
        Args:
            experiment_name: Tên thí nghiệm
            metrics_values: Dict chứa {tên_số_liệu: giá_trị}
            step: Bước hiện tại
            
        Returns:
            Dict chứa kết quả cập nhật cho từng số liệu
        """
        results = {}
        
        for metric_name, value in metrics_values.items():
            results[metric_name] = self.update_metric(experiment_name, metric_name, value, step)
        
        return results
    
    def remove_metric(self, experiment_name: str, metric_name: str) -> bool:
        """
        Xóa một số liệu.
        
        Args:
            experiment_name: Tên thí nghiệm
            metric_name: Tên số liệu
            
        Returns:
            True nếu xóa thành công, False nếu không
        """
        if experiment_name in self.metrics and metric_name in self.metrics[experiment_name]:
            del self.metrics[experiment_name][metric_name]
            return True
        return False
    
    def reset_experiment(self, experiment_name: str) -> bool:
        """
        Đặt lại tất cả số liệu trong một thí nghiệm.
        
        Args:
            experiment_name: Tên thí nghiệm
            
        Returns:
            True nếu đặt lại thành công, False nếu không
        """
        if experiment_name in self.metrics:
            for metric in self.metrics[experiment_name].values():
                metric.reset()
            return True
        return False
    
    def export_metrics(self, experiment_name: str, output_path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """
        Xuất số liệu của một thí nghiệm ra file JSON.
        
        Args:
            experiment_name: Tên thí nghiệm
            output_path: Đường dẫn file xuất (None để tạo tự động)
            
        Returns:
            Đường dẫn file đã xuất hoặc None nếu có lỗi
        """
        if experiment_name not in self.metrics:
            self.logger.warning(f"Không tìm thấy thí nghiệm '{experiment_name}'")
            return None
        
        # Tạo đường dẫn tự động nếu không được cung cấp
        if output_path is None:
            output_path = self.root_log_dir / f"{experiment_name}_metrics_export.json"
        else:
            output_path = Path(output_path)
        
        try:
            # Chuẩn bị dữ liệu để xuất
            export_data = {
                "experiment_name": experiment_name,
                "export_time": datetime.now().isoformat(),
                "metrics": {}
            }
            
            # Thêm dữ liệu cho từng số liệu
            for metric_name, metric in self.metrics[experiment_name].items():
                values, steps = metric.get_history()
                
                # Chuyển đổi numpy arrays thành lists
                if isinstance(values, list) and len(values) > 0:
                    if isinstance(values[0], np.ndarray):
                        processed_values = [v.tolist() for v in values]
                    elif isinstance(values[0], (np.int32, np.int64, np.float32, np.float64)):
                        processed_values = [float(v) for v in values]
                    else:
                        processed_values = values
                else:
                    processed_values = values
                
                export_data["metrics"][metric_name] = {
                    "description": metric.description,
                    "values": processed_values,
                    "steps": steps,
                    "last_value": metric.last_value if not isinstance(metric.last_value, np.ndarray) else metric.last_value.tolist(),
                    "last_step": metric.last_step
                }
            
            # Lưu vào file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=4, ensure_ascii=False, default=str)
            
            self.logger.info(f"Đã xuất số liệu thí nghiệm '{experiment_name}' vào {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Lỗi khi xuất số liệu thí nghiệm '{experiment_name}': {str(e)}")
            return None
    
    def import_metrics(self, input_path: Union[str, Path], experiment_name: Optional[str] = None) -> Optional[str]:
        """
        Nhập số liệu từ file JSON.
        
        Args:
            input_path: Đường dẫn file nhập
            experiment_name: Tên thí nghiệm (None để lấy từ file)
            
        Returns:
            Tên thí nghiệm đã nhập hoặc None nếu có lỗi
        """
        try:
            # Đọc file
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Lấy tên thí nghiệm
            file_experiment_name = data.get("experiment_name")
            if experiment_name is None:
                experiment_name = file_experiment_name
            
            if experiment_name is None:
                self.logger.error("Không tìm thấy tên thí nghiệm trong file và không được cung cấp")
                return None
            
            # Xóa metrics hiện có của thí nghiệm nếu có
            if experiment_name in self.metrics:
                self.metrics[experiment_name] = {}
            
            # Nhập từng số liệu
            for metric_name, metric_data in data.get("metrics", {}).items():
                # Xác định loại số liệu và tạo đối tượng tương ứng
                values = metric_data.get("values", [])
                description = metric_data.get("description", "")
                
                # Kiểm tra loại dữ liệu đầu tiên để quyết định loại số liệu
                if len(values) > 0:
                    first_value = values[0]
                    if isinstance(first_value, (int, float)):
                        metric = self.create_scalar_metric(
                            name=metric_name,
                            description=description,
                            experiment_name=experiment_name
                        )
                    elif isinstance(first_value, list) and all(isinstance(x, (int, float)) for x in first_value):
                        metric = self.create_histogram_metric(
                            name=metric_name,
                            description=description,
                            experiment_name=experiment_name
                        )
                    elif isinstance(first_value, dict):
                        metric = self.create_hyperparameter_metric(
                            name=metric_name,
                            description=description,
                            experiment_name=experiment_name
                        )
                    elif isinstance(first_value, str):
                        metric = self.create_text_metric(
                            name=metric_name,
                            description=description,
                            experiment_name=experiment_name
                        )
                    else:
                        # Mặc định là scalar nếu không xác định được
                        metric = self.create_scalar_metric(
                            name=metric_name,
                            description=description,
                            experiment_name=experiment_name
                        )
                else:
                    # Tạo scalar nếu không có dữ liệu
                    metric = self.create_scalar_metric(
                        name=metric_name,
                        description=description,
                        experiment_name=experiment_name
                    )
                
                # Thêm dữ liệu vào số liệu
                steps = metric_data.get("steps", [])
                for i, (value, step) in enumerate(zip(values, steps)):
                    metric.update(value, step)
                
                self.logger.debug(f"Đã nhập số liệu '{metric_name}' với {len(values)} giá trị")
            
            self.logger.info(f"Đã nhập số liệu cho thí nghiệm '{experiment_name}' từ {input_path}")
            return experiment_name
            
        except Exception as e:
            self.logger.error(f"Lỗi khi nhập số liệu: {str(e)}")
            return None


# Hàm tiện ích

def create_metrics_for_agent(
    agent_name: str,
    env_name: str,
    experiment_name: Optional[str] = None,
    root_log_dir: Optional[Union[str, Path]] = None,
    agent_type: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> MetricsManager:
    """
    Tạo bộ số liệu đầy đủ cho việc huấn luyện agent.
    
    Args:
        agent_name: Tên agent
        env_name: Tên môi trường
        experiment_name: Tên thí nghiệm (None để tạo tự động)
        root_log_dir: Thư mục gốc cho log
        agent_type: Loại agent (DQN, PPO, v.v.)
        logger: Logger tùy chỉnh
        
    Returns:
        Đối tượng MetricsManager với các số liệu đã tạo
    """
    # Tạo tên thí nghiệm nếu không được cung cấp
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{agent_name}_{env_name}_{timestamp}"
    
    # Khởi tạo manager
    manager = MetricsManager(root_log_dir=root_log_dir, logger=logger)
    
    # Tạo các số liệu cơ bản
    manager.create_scalar_metric(
        name="episode_reward",
        description="Phần thưởng nhận được trong mỗi episode",
        experiment_name=experiment_name
    )
    
    manager.create_scalar_metric(
        name="episode_length",
        description="Số bước trong mỗi episode",
        experiment_name=experiment_name
    )
    
    manager.create_scalar_metric(
        name="loss",
        description="Giá trị loss trong quá trình huấn luyện",
        experiment_name=experiment_name
    )
    
    manager.create_scalar_metric(
        name="epsilon",
        description="Giá trị epsilon cho chiến lược khám phá",
        experiment_name=experiment_name
    )
    
    manager.create_scalar_metric(
        name="learning_rate",
        description="Tốc độ học",
        experiment_name=experiment_name
    )
    
    # Tạo các số liệu đặc thù cho từng loại agent
    if agent_type is not None:
        if agent_type.lower() in ["ppo", "a2c"]:
            manager.create_scalar_metric(
                name="entropy",
                description="Entropy của policy",
                experiment_name=experiment_name
            )
            
            manager.create_scalar_metric(
                name="kl_divergence",
                description="KL divergence giữa policy cũ và mới",
                experiment_name=experiment_name
            )
            
            manager.create_scalar_metric(
                name="value_loss",
                description="Loss của value function",
                experiment_name=experiment_name
            )
            
            manager.create_scalar_metric(
                name="policy_loss",
                description="Loss của policy function",
                experiment_name=experiment_name
            )
        
        elif agent_type.lower() == "dqn":
            manager.create_scalar_metric(
                name="q_values/mean",
                description="Giá trị Q trung bình",
                experiment_name=experiment_name
            )
            
            manager.create_scalar_metric(
                name="q_values/max",
                description="Giá trị Q tối đa",
                experiment_name=experiment_name
            )
            
            manager.create_scalar_metric(
                name="target_updates",
                description="Số lần cập nhật mạng target",
                experiment_name=experiment_name
            )
    
    # Các số liệu thống kê hiệu suất
    manager.create_scalar_metric(
        name="win_rate",
        description="Tỷ lệ thắng",
        experiment_name=experiment_name
    )
    
    manager.create_scalar_metric(
        name="avg_return",
        description="Lợi nhuận trung bình",
        experiment_name=experiment_name
    )
    
    # Các siêu tham số
    manager.create_hyperparameter_metric(
        name="hyperparameters",
        description="Siêu tham số huấn luyện",
        experiment_name=experiment_name
    )
    
    # Actions distribution
    manager.create_histogram_metric(
        name="actions_distribution",
        description="Phân phối các hành động được chọn",
        experiment_name=experiment_name
    )
    
    # Thêm text summary
    manager.create_text_metric(
        name="training_summary",
        description="Tóm tắt quá trình huấn luyện",
        experiment_name=experiment_name
    )
    
    return manager


def create_metrics_for_trading(
    strategy_name: str,
    exchange_name: str,
    symbol: str,
    timeframe: str,
    experiment_name: Optional[str] = None,
    root_log_dir: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None
) -> MetricsManager:
    """
    Tạo bộ số liệu đầy đủ cho việc giao dịch.
    
    Args:
        strategy_name: Tên chiến lược
        exchange_name: Tên sàn giao dịch
        symbol: Ký hiệu cặp tiền
        timeframe: Khung thời gian
        experiment_name: Tên thí nghiệm (None để tạo tự động)
        root_log_dir: Thư mục gốc cho log
        logger: Logger tùy chỉnh
        
    Returns:
        Đối tượng MetricsManager với các số liệu đã tạo
    """
    # Tạo tên thí nghiệm nếu không được cung cấp
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{strategy_name}_{symbol}_{timeframe}_{timestamp}"
    
    # Khởi tạo manager
    manager = MetricsManager(root_log_dir=root_log_dir, logger=logger)
    
    # Các số liệu lợi nhuận
    manager.create_scalar_metric(
        name="total_profit",
        description="Tổng lợi nhuận",
        experiment_name=experiment_name
    )
    
    manager.create_scalar_metric(
        name="cumulative_return",
        description="Lợi nhuận tích lũy",
        experiment_name=experiment_name
    )
    
    manager.create_scalar_metric(
        name="daily_profit",
        description="Lợi nhuận hàng ngày",
        experiment_name=experiment_name
    )
    
    # Các số liệu giao dịch
    manager.create_scalar_metric(
        name="win_rate",
        description="Tỷ lệ thắng",
        experiment_name=experiment_name
    )
    
    manager.create_scalar_metric(
        name="total_trades",
        description="Tổng số giao dịch",
        experiment_name=experiment_name
    )
    
    manager.create_scalar_metric(
        name="winning_trades",
        description="Số giao dịch thắng",
        experiment_name=experiment_name
    )
    
    manager.create_scalar_metric(
        name="losing_trades",
        description="Số giao dịch thua",
        experiment_name=experiment_name
    )
    
    # Các số liệu rủi ro
    manager.create_scalar_metric(
        name="max_drawdown",
        description="Sụt giảm vốn tối đa",
        experiment_name=experiment_name
    )
    
    manager.create_scalar_metric(
        name="sharpe_ratio",
        description="Tỷ số Sharpe",
        experiment_name=experiment_name
    )
    
    manager.create_scalar_metric(
        name="sortino_ratio",
        description="Tỷ số Sortino",
        experiment_name=experiment_name
    )
    
    manager.create_scalar_metric(
        name="volatility",
        description="Độ biến động lợi nhuận",
        experiment_name=experiment_name
    )
    
    # Trade details
    manager.create_text_metric(
        name="recent_trades",
        description="Chi tiết các giao dịch gần đây",
        experiment_name=experiment_name
    )
    
    # Phân phối lợi nhuận
    manager.create_histogram_metric(
        name="profit_distribution",
        description="Phân phối lợi nhuận",
        experiment_name=experiment_name
    )
    
    # Siêu tham số
    manager.create_hyperparameter_metric(
        name="strategy_params",
        description="Tham số chiến lược",
        experiment_name=experiment_name
    )
    
    # Tóm tắt giao dịch
    manager.create_text_metric(
        name="trading_summary",
        description="Tóm tắt hiệu suất giao dịch",
        experiment_name=experiment_name
    )
    
    return manager


def log_model_structure(
    model: tf.keras.Model,
    metrics_manager: MetricsManager,
    experiment_name: str,
    include_weights: bool = False,
    step: int = 0
) -> None:
    """
    Ghi cấu trúc mô hình vào TensorBoard.
    
    Args:
        model: Mô hình Keras
        metrics_manager: Đối tượng quản lý số liệu
        experiment_name: Tên thí nghiệm
        include_weights: Có bao gồm trọng số không
        step: Bước hiện tại
    """
    try:
        # Tạo metric nếu chưa có
        model_text_metric = metrics_manager.get_metric(experiment_name, "model_structure")
        if model_text_metric is None:
            model_text_metric = metrics_manager.create_text_metric(
                name="model_structure",
                description="Cấu trúc mô hình",
                experiment_name=experiment_name
            )
        
        # Tạo chuỗi mô tả mô hình
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        
        if include_weights:
            # Thêm thông tin về các trọng số
            weight_info = []
            for layer in model.layers:
                weights = layer.get_weights()
                if weights:
                    weight_shapes = [w.shape for w in weights]
                    weight_info.append(f"Layer: {layer.name}, Weights: {weight_shapes}")
            
            model_summary += "\n\nWeight Info:\n" + "\n".join(weight_info)
        
        # Cập nhật số liệu
        model_text_metric.update(model_summary, step)
        
    except Exception as e:
        logger = get_logger("tensorboard_metrics")
        logger.error(f"Lỗi khi ghi cấu trúc mô hình: {str(e)}")


def log_backtest_results(
    results: Dict[str, Any],
    metrics_manager: MetricsManager,
    experiment_name: str,
    step: int = 0
) -> None:
    """
    Ghi kết quả backtest vào TensorBoard.
    
    Args:
        results: Dict chứa kết quả backtest
        metrics_manager: Đối tượng quản lý số liệu
        experiment_name: Tên thí nghiệm
        step: Bước hiện tại
    """
    try:
        # Tạo metric cho backtest summary nếu chưa có
        backtest_summary_metric = metrics_manager.get_metric(experiment_name, "backtest_summary")
        if backtest_summary_metric is None:
            backtest_summary_metric = metrics_manager.create_text_metric(
                name="backtest_summary",
                description="Tóm tắt kết quả backtest",
                experiment_name=experiment_name
            )
        
        # Tạo các metric scalar cho các chỉ số backtest chính
        metrics_to_log = {
            "total_return": "Tổng lợi nhuận",
            "annualized_return": "Lợi nhuận hàng năm",
            "sharpe_ratio": "Tỷ số Sharpe",
            "sortino_ratio": "Tỷ số Sortino",
            "max_drawdown": "Sụt giảm vốn tối đa",
            "win_rate": "Tỷ lệ thắng",
            "profit_factor": "Hệ số lợi nhuận",
            "expectancy": "Kỳ vọng",
            "average_trade": "Giao dịch trung bình",
            "average_win": "Thắng trung bình",
            "average_loss": "Thua trung bình"
        }
        
        # Cập nhật từng metric
        for metric_key, metric_desc in metrics_to_log.items():
            if metric_key in results:
                # Lấy giá trị
                value = results[metric_key]
                
                # Lấy hoặc tạo metric
                metric = metrics_manager.get_metric(experiment_name, metric_key)
                if metric is None:
                    metric = metrics_manager.create_scalar_metric(
                        name=metric_key,
                        description=metric_desc,
                        experiment_name=experiment_name
                    )
                
                # Cập nhật giá trị
                metric.update(value, step)
        
        # Tạo tóm tắt văn bản
        backtest_text = f"Backtest Results (Step {step}):\n\n"
        
        for metric_key, metric_desc in metrics_to_log.items():
            if metric_key in results:
                value = results[metric_key]
                backtest_text += f"{metric_desc}: {value}\n"
        
        # Thêm thông tin bổ sung nếu có
        if "trades" in results:
            num_trades = len(results["trades"])
            backtest_text += f"\nTotal Trades: {num_trades}\n"
        
        if "duration" in results:
            backtest_text += f"Backtest Duration: {results['duration']}\n"
        
        # Cập nhật tóm tắt
        backtest_summary_metric.update(backtest_text, step)
        
    except Exception as e:
        logger = get_logger("tensorboard_metrics")
        logger.error(f"Lỗi khi ghi kết quả backtest: {str(e)}")


def log_predictions_vs_reality(
    predictions: np.ndarray,
    actual_values: np.ndarray,
    metrics_manager: MetricsManager,
    experiment_name: str,
    step: int = 0,
    max_points: int = 100
) -> None:
    """
    Ghi so sánh giữa dự đoán và giá trị thực tế vào TensorBoard.
    
    Args:
        predictions: Mảng các giá trị dự đoán
        actual_values: Mảng các giá trị thực tế
        metrics_manager: Đối tượng quản lý số liệu
        experiment_name: Tên thí nghiệm
        step: Bước hiện tại
        max_points: Số điểm tối đa hiển thị
    """
    try:
        import matplotlib.pyplot as plt
        import io
        
        # Tạo metric nếu chưa có
        pred_vs_real_metric = metrics_manager.get_metric(experiment_name, "predictions_vs_reality")
        if pred_vs_real_metric is None:
            pred_vs_real_metric = metrics_manager.create_image_metric(
                name="predictions_vs_reality",
                description="So sánh giữa dự đoán và thực tế",
                experiment_name=experiment_name
            )
        
        # Giới hạn số điểm hiển thị
        if len(predictions) > max_points:
            # Lấy mẫu ngẫu nhiên
            indices = np.random.choice(len(predictions), max_points, replace=False)
            indices = np.sort(indices)
            predictions = predictions[indices]
            actual_values = actual_values[indices]
        
        # Tạo biểu đồ
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Vẽ đường y=x làm tham chiếu
        min_val = min(np.min(predictions), np.min(actual_values))
        max_val = max(np.max(predictions), np.max(actual_values))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        
        # Vẽ dữ liệu
        ax.scatter(actual_values, predictions, alpha=0.5)
        
        # Thiết lập nhãn
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predictions')
        ax.set_title('Predictions vs Actual Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Thêm thông tin về lỗi
        mse = np.mean((predictions - actual_values) ** 2)
        mae = np.mean(np.abs(predictions - actual_values))
        ax.text(0.05, 0.95, f'MSE: {mse:.4f}\nMAE: {mae:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Chuyển đổi biểu đồ thành hình ảnh
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        
        # Đọc PNG thành tensor
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        
        # Cập nhật metric
        pred_vs_real_metric.update(image, step)
        
        # Cập nhật các metric lỗi
        mse_metric = metrics_manager.get_metric(experiment_name, "mse")
        if mse_metric is None:
            mse_metric = metrics_manager.create_scalar_metric(
                name="mse",
                description="Mean Squared Error",
                experiment_name=experiment_name
            )
        mse_metric.update(mse, step)
        
        mae_metric = metrics_manager.get_metric(experiment_name, "mae")
        if mae_metric is None:
            mae_metric = metrics_manager.create_scalar_metric(
                name="mae",
                description="Mean Absolute Error",
                experiment_name=experiment_name
            )
        mae_metric.update(mae, step)
        
    except Exception as e:
        logger = get_logger("tensorboard_metrics")
        logger.error(f"Lỗi khi ghi so sánh dự đoán và thực tế: {str(e)}")