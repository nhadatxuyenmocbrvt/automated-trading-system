"""
Chỉ số hiệu quả cho quá trình tự động hóa.
File này định nghĩa các lớp và hàm để đo lường hiệu quả của quá trình huấn luyện,
sử dụng tài nguyên hệ thống, và thời gian thực thi để tối ưu hóa việc tái huấn luyện.
"""

import numpy as np
import pandas as pd
import psutil
import time
import json
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import os

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config

class EfficiencyMetrics:
    """
    Lớp tính toán các chỉ số hiệu quả của quá trình huấn luyện và tái huấn luyện.
    Bao gồm các chỉ số về thời gian, sử dụng tài nguyên và hiệu quả huấn luyện.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        efficiency_threshold: float = 0.8,
        resource_threshold: float = 0.7,
        time_threshold: float = 1.5,
        history_file: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo đối tượng EfficiencyMetrics.
        
        Args:
            config (Dict[str, Any], optional): Cấu hình tùy chỉnh
            efficiency_threshold (float, optional): Ngưỡng hiệu quả tối thiểu (0.0-1.0)
            resource_threshold (float, optional): Ngưỡng sử dụng tài nguyên tối đa (0.0-1.0)
            time_threshold (float, optional): Ngưỡng thời gian tối đa (hệ số so với lần trước)
            history_file (Union[str, Path], optional): File lưu lịch sử chỉ số
            logger (logging.Logger, optional): Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("efficiency_metrics")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config() if config is None else config
        
        # Thiết lập ngưỡng
        self.efficiency_threshold = efficiency_threshold
        self.resource_threshold = resource_threshold
        self.time_threshold = time_threshold
        
        # Đường dẫn lưu lịch sử
        self.history_file = Path(history_file) if history_file else None
        
        # Khởi tạo các biến theo dõi hiệu quả
        self.training_history = {
            "timestamps": [],
            "training_time": [],
            "iterations": [],
            "loss_values": [],
            "cpu_usage": [],
            "memory_usage": [],
            "gpu_usage": [],
            "efficiency_scores": [],
            "hyperparameters": []
        }
        
        # Khởi tạo theo dõi tài nguyên
        self.resource_tracking = {
            "timestamps": [],
            "cpu_usage": [],
            "memory_usage": [],
            "gpu_usage": [],
            "disk_usage": []
        }
        
        self.baseline_metrics = None
        self._current_session_start = None
        self._resource_tracking_interval = 5  # Giây
        self._is_tracking_resources = False
        
        self.logger.info("Đã khởi tạo EfficiencyMetrics")
    
    def start_session(self, session_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Bắt đầu một phiên theo dõi hiệu quả mới.
        
        Args:
            session_info (Dict[str, Any], optional): Thông tin bổ sung về phiên
        """
        self._current_session_start = datetime.now()
        self._current_session_info = session_info or {}
        
        # Bắt đầu theo dõi tài nguyên
        self._start_resource_tracking()
        
        session_name = self._current_session_info.get("name", "Phiên huấn luyện")
        self.logger.info(f"Bắt đầu phiên theo dõi hiệu quả: {session_name} ({self._current_session_start})")
    
    def end_session(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Kết thúc phiên theo dõi và tính toán các chỉ số hiệu quả.
        
        Args:
            results (Dict[str, Any]): Kết quả huấn luyện và các thông tin bổ sung
            
        Returns:
            Dict[str, Any]: Các chỉ số hiệu quả đã tính toán
        """
        if self._current_session_start is None:
            self.logger.warning("Không có phiên nào đang diễn ra để kết thúc")
            return {}
        
        # Dừng theo dõi tài nguyên
        self._stop_resource_tracking()
        
        # Tính toán thời gian huấn luyện
        end_time = datetime.now()
        training_time = (end_time - self._current_session_start).total_seconds()
        
        # Tính toán các chỉ số hiệu quả
        efficiency_metrics = self._calculate_efficiency(training_time, results)
        
        # Cập nhật lịch sử
        self._update_history(efficiency_metrics)
        
        # Lưu lịch sử nếu có file
        if self.history_file:
            self.save_history(self.history_file)
        
        self.logger.info(f"Kết thúc phiên theo dõi hiệu quả ({end_time})")
        self.logger.info(f"Thời gian huấn luyện: {training_time:.2f}s, Hiệu quả: {efficiency_metrics['efficiency_score']:.4f}")
        
        # Đặt lại biến phiên hiện tại
        self._current_session_start = None
        
        return efficiency_metrics
    
    def _calculate_efficiency(self, training_time: float, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tính toán các chỉ số hiệu quả từ kết quả huấn luyện.
        
        Args:
            training_time (float): Thời gian huấn luyện (giây)
            results (Dict[str, Any]): Kết quả huấn luyện
            
        Returns:
            Dict[str, Any]: Các chỉ số hiệu quả
        """
        # Lấy các thông số từ kết quả
        epochs = results.get("epochs", 0)
        iterations = results.get("iterations", 0)
        final_loss = results.get("final_loss", float('inf'))
        loss_history = results.get("loss_history", [])
        
        # Tính toán sử dụng tài nguyên trung bình
        avg_cpu_usage = np.mean(self.resource_tracking["cpu_usage"]) if self.resource_tracking["cpu_usage"] else 0
        avg_memory_usage = np.mean(self.resource_tracking["memory_usage"]) if self.resource_tracking["memory_usage"] else 0
        avg_gpu_usage = np.mean(self.resource_tracking["gpu_usage"]) if self.resource_tracking["gpu_usage"] else 0
        
        # Tính hội tụ
        convergence_rate = 0.0
        if len(loss_history) >= 2:
            # Tính tốc độ giảm loss
            loss_changes = [(loss_history[i-1] - loss_history[i]) / max(loss_history[i-1], 1e-10) 
                           for i in range(1, len(loss_history))]
            convergence_rate = np.mean(loss_changes) if loss_changes else 0.0
        
        # Tính toán thời gian trên mỗi epoch/iteration
        time_per_epoch = training_time / max(epochs, 1)
        time_per_iteration = training_time / max(iterations, 1)
        
        # Tính điểm hiệu quả tổng thể
        # Công thức: kết hợp tốc độ hội tụ và hiệu quả tài nguyên
        efficiency_score = self._compute_efficiency_score(
            convergence_rate=convergence_rate,
            time_per_epoch=time_per_epoch,
            cpu_usage=avg_cpu_usage,
            memory_usage=avg_memory_usage,
            gpu_usage=avg_gpu_usage,
            final_loss=final_loss
        )
        
        # So sánh với baseline nếu có
        baseline_comparison = {}
        if self.baseline_metrics is not None:
            # Tính % thay đổi so với baseline
            time_change = (time_per_epoch / self.baseline_metrics["time_per_epoch"]) - 1.0 if self.baseline_metrics["time_per_epoch"] > 0 else 0.0
            loss_change = (final_loss / self.baseline_metrics["final_loss"]) - 1.0 if self.baseline_metrics["final_loss"] > 0 else 0.0
            efficiency_change = (efficiency_score / self.baseline_metrics["efficiency_score"]) - 1.0 if self.baseline_metrics["efficiency_score"] > 0 else 0.0
            
            baseline_comparison = {
                "time_change": time_change,
                "loss_change": loss_change,
                "efficiency_change": efficiency_change,
                "is_better": efficiency_change > 0  # True nếu hiệu quả tốt hơn baseline
            }
        
        # Tập hợp kết quả
        efficiency_metrics = {
            "timestamp": datetime.now(),
            "training_time": training_time,
            "epochs": epochs,
            "iterations": iterations,
            "final_loss": final_loss,
            "convergence_rate": convergence_rate,
            "time_per_epoch": time_per_epoch,
            "time_per_iteration": time_per_iteration,
            "avg_cpu_usage": avg_cpu_usage,
            "avg_memory_usage": avg_memory_usage,
            "avg_gpu_usage": avg_gpu_usage,
            "efficiency_score": efficiency_score,
            "baseline_comparison": baseline_comparison,
            "hyperparameters": results.get("hyperparameters", {})
        }
        
        return efficiency_metrics
    
    def _compute_efficiency_score(
        self,
        convergence_rate: float,
        time_per_epoch: float,
        cpu_usage: float,
        memory_usage: float,
        gpu_usage: float,
        final_loss: float,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Tính toán điểm hiệu quả tổng thể từ các thành phần.
        
        Args:
            convergence_rate (float): Tốc độ hội tụ
            time_per_epoch (float): Thời gian trên mỗi epoch
            cpu_usage (float): Mức sử dụng CPU
            memory_usage (float): Mức sử dụng bộ nhớ
            gpu_usage (float): Mức sử dụng GPU
            final_loss (float): Giá trị loss cuối cùng
            weights (Dict[str, float], optional): Trọng số cho từng thành phần
            
        Returns:
            float: Điểm hiệu quả (0.0-1.0)
        """
        # Trọng số mặc định
        if weights is None:
            weights = {
                "convergence_rate": 0.3,
                "time_per_epoch": 0.2,
                "resource_usage": 0.2,
                "final_loss": 0.3
            }
        
        # Chuẩn hóa các thành phần
        
        # 1. Tốc độ hội tụ: cao hơn là tốt hơn
        convergence_score = min(max(convergence_rate, 0.0), 1.0)
        
        # 2. Thời gian trên mỗi epoch: thấp hơn là tốt hơn
        # So sánh với baseline nếu có, nếu không thì sử dụng thang đo tương đối
        if self.baseline_metrics is not None and self.baseline_metrics["time_per_epoch"] > 0:
            time_score = min(self.baseline_metrics["time_per_epoch"] / max(time_per_epoch, 1e-10), 2.0) / 2.0
        else:
            # Giả sử 10s là ngưỡng dưới lý tưởng, 300s là ngưỡng trên
            time_score = max(0.0, min(1.0, 1.0 - (time_per_epoch - 10) / 290))
        
        # 3. Mức sử dụng tài nguyên: tối ưu ở mức vừa phải (không quá cao, không quá thấp)
        # Tối ưu khoảng 60-80%
        cpu_score = 1.0 - abs(cpu_usage - 0.7) / 0.7
        memory_score = 1.0 - abs(memory_usage - 0.7) / 0.7
        gpu_score = 1.0 - abs(gpu_usage - 0.7) / 0.7 if gpu_usage > 0 else 0.0
        
        # Kết hợp các điểm tài nguyên
        resource_score = (cpu_score + memory_score + (gpu_score if gpu_usage > 0 else 0)) / (3 if gpu_usage > 0 else 2)
        
        # 4. Giá trị loss cuối cùng: thấp hơn là tốt hơn
        # So sánh với baseline nếu có
        if self.baseline_metrics is not None and self.baseline_metrics["final_loss"] > 0:
            loss_score = min(self.baseline_metrics["final_loss"] / max(final_loss, 1e-10), 2.0) / 2.0
        else:
            # Nếu không có baseline, chuẩn hóa theo biên độ
            loss_score = max(0.0, min(1.0, 1.0 - final_loss / 10.0))
        
        # Tính điểm cuối cùng
        efficiency_score = (
            weights["convergence_rate"] * convergence_score +
            weights["time_per_epoch"] * time_score +
            weights["resource_usage"] * resource_score +
            weights["final_loss"] * loss_score
        )
        
        return max(0.0, min(1.0, efficiency_score))
    
    def _update_history(self, metrics: Dict[str, Any]) -> None:
        """
        Cập nhật lịch sử với chỉ số mới.
        
        Args:
            metrics (Dict[str, Any]): Chỉ số hiệu quả mới
        """
        # Cập nhật lịch sử huấn luyện
        self.training_history["timestamps"].append(metrics["timestamp"])
        self.training_history["training_time"].append(metrics["training_time"])
        self.training_history["iterations"].append(metrics["iterations"])
        self.training_history["loss_values"].append(metrics["final_loss"])
        self.training_history["cpu_usage"].append(metrics["avg_cpu_usage"])
        self.training_history["memory_usage"].append(metrics["avg_memory_usage"])
        self.training_history["gpu_usage"].append(metrics["avg_gpu_usage"])
        self.training_history["efficiency_scores"].append(metrics["efficiency_score"])
        self.training_history["hyperparameters"].append(metrics["hyperparameters"])
        
        # Đặt baseline nếu chưa có
        if self.baseline_metrics is None:
            self.set_baseline(metrics)
    
    def _start_resource_tracking(self) -> None:
        """
        Bắt đầu theo dõi tài nguyên hệ thống.
        """
        # Đặt lại tracking
        self.resource_tracking = {
            "timestamps": [],
            "cpu_usage": [],
            "memory_usage": [],
            "gpu_usage": [],
            "disk_usage": []
        }
        
        self._is_tracking_resources = True
        
        # Bắt đầu theo dõi tài nguyên trong một luồng riêng
        # Trong trường hợp này, chúng ta mô phỏng việc theo dõi bằng cách lấy mẫu định kỳ
        self._get_resource_usage()
    
    def _stop_resource_tracking(self) -> None:
        """
        Dừng theo dõi tài nguyên hệ thống.
        """
        self._is_tracking_resources = False
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """
        Lấy thông tin sử dụng tài nguyên hệ thống hiện tại.
        
        Returns:
            Dict[str, float]: Thông tin tài nguyên
        """
        # Lấy mẫu tài nguyên hiện tại
        cpu_percent = psutil.cpu_percent() / 100.0
        memory_percent = psutil.virtual_memory().percent / 100.0
        disk_percent = psutil.disk_usage('/').percent / 100.0
        
        # Mô phỏng GPU (thực tế cần sử dụng thư viện như nvidia-ml-py hoặc pynvml)
        gpu_percent = 0.0
        try:
            # Mô phỏng lấy thông tin GPU
            # Trong thực tế, cần sử dụng NVIDIA API hoặc thư viện tương tự
            gpu_percent = np.random.uniform(0.5, 0.9)
        except:
            pass
        
        # Cập nhật lịch sử tracking
        now = datetime.now()
        self.resource_tracking["timestamps"].append(now)
        self.resource_tracking["cpu_usage"].append(cpu_percent)
        self.resource_tracking["memory_usage"].append(memory_percent)
        self.resource_tracking["gpu_usage"].append(gpu_percent)
        self.resource_tracking["disk_usage"].append(disk_percent)
        
        # Sắp xếp lại việc lấy mẫu tiếp theo nếu đang theo dõi
        if self._is_tracking_resources:
            # Trong môi trường thực tế, nên sử dụng threading hoặc asyncio
            # Ở đây mô phỏng bằng cách lấy mẫu định kỳ
            time.sleep(self._resource_tracking_interval)
            self._get_resource_usage()
        
        return {
            "cpu": cpu_percent,
            "memory": memory_percent,
            "gpu": gpu_percent,
            "disk": disk_percent
        }
    
    def set_baseline(self, metrics: Dict[str, Any]) -> None:
        """
        Đặt baseline cho so sánh hiệu quả.
        
        Args:
            metrics (Dict[str, Any]): Chỉ số hiệu quả để sử dụng làm baseline
        """
        self.baseline_metrics = {
            "timestamp": metrics["timestamp"],
            "training_time": metrics["training_time"],
            "time_per_epoch": metrics.get("time_per_epoch", metrics["training_time"] / max(metrics.get("epochs", 1), 1)),
            "final_loss": metrics["final_loss"],
            "efficiency_score": metrics["efficiency_score"],
            "avg_cpu_usage": metrics["avg_cpu_usage"],
            "avg_memory_usage": metrics["avg_memory_usage"],
            "avg_gpu_usage": metrics["avg_gpu_usage"],
            "hyperparameters": metrics.get("hyperparameters", {})
        }
        
        self.logger.info(f"Đã đặt baseline hiệu quả từ phiên {metrics['timestamp']}")
    
    def reset_baseline(self) -> None:
        """
        Đặt lại baseline.
        """
        self.baseline_metrics = None
        self.logger.info("Đã đặt lại baseline hiệu quả")
    
    def should_early_stop(
        self, 
        current_metrics: Dict[str, Any],
        min_epochs: int = 5
    ) -> Tuple[bool, str]:
        """
        Xác định xem có nên dừng sớm quá trình huấn luyện dựa trên hiệu quả không.
        
        Args:
            current_metrics (Dict[str, Any]): Chỉ số hiệu quả hiện tại
            min_epochs (int, optional): Số epochs tối thiểu trước khi xem xét dừng sớm
            
        Returns:
            Tuple[bool, str]: (Nên dừng sớm?, Lý do)
        """
        if current_metrics.get("epochs", 0) < min_epochs:
            return False, "Chưa đạt số epochs tối thiểu"
        
        reasons = []
        
        # 1. Nếu hiệu quả quá thấp
        if current_metrics.get("efficiency_score", 1.0) < self.efficiency_threshold:
            reasons.append(f"Hiệu quả quá thấp ({current_metrics['efficiency_score']:.4f} < {self.efficiency_threshold})")
        
        # 2. Nếu sử dụng tài nguyên quá cao
        if current_metrics.get("avg_cpu_usage", 0) > self.resource_threshold:
            reasons.append(f"Sử dụng CPU quá cao ({current_metrics['avg_cpu_usage']:.2%} > {self.resource_threshold:.2%})")
        
        if current_metrics.get("avg_memory_usage", 0) > self.resource_threshold:
            reasons.append(f"Sử dụng Memory quá cao ({current_metrics['avg_memory_usage']:.2%} > {self.resource_threshold:.2%})")
        
        # 3. Nếu thời gian quá lâu so với baseline
        if self.baseline_metrics is not None:
            time_ratio = current_metrics.get("time_per_epoch", float('inf')) / max(self.baseline_metrics["time_per_epoch"], 1e-10)
            if time_ratio > self.time_threshold:
                reasons.append(f"Thời gian huấn luyện quá lâu (gấp {time_ratio:.2f} lần baseline)")
        
        # Quyết định dừng sớm nếu có ít nhất một lý do
        should_stop = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "Không có vấn đề về hiệu quả"
        
        if should_stop:
            self.logger.warning(f"Đề xuất dừng sớm huấn luyện: {reason}")
        
        return should_stop, reason
    
    def compare_hyperparameters(
        self,
        metrics1: Dict[str, Any],
        metrics2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        So sánh hiệu quả giữa hai tập hyperparameters.
        
        Args:
            metrics1 (Dict[str, Any]): Chỉ số hiệu quả với tập hyperparameters 1
            metrics2 (Dict[str, Any]): Chỉ số hiệu quả với tập hyperparameters 2
            
        Returns:
            Dict[str, Any]: Kết quả so sánh
        """
        # So sánh các chỉ số chính
        efficiency_change = (metrics2["efficiency_score"] / max(metrics1["efficiency_score"], 1e-10)) - 1.0
        loss_change = (metrics1["final_loss"] / max(metrics2["final_loss"], 1e-10)) - 1.0  # Ngược lại vì loss thấp hơn là tốt hơn
        time_change = (metrics1["training_time"] / max(metrics2["training_time"], 1e-10)) - 1.0  # Ngược lại vì thời gian ngắn hơn là tốt hơn
        
        # So sánh hyperparameters
        hyperparam_diff = {}
        for key in set(metrics1.get("hyperparameters", {}).keys()).union(set(metrics2.get("hyperparameters", {}).keys())):
            value1 = metrics1.get("hyperparameters", {}).get(key, None)
            value2 = metrics2.get("hyperparameters", {}).get(key, None)
            if value1 != value2:
                hyperparam_diff[key] = {"from": value1, "to": value2}
        
        # Tính giá trị cải thiện tổng thể
        # Công thức: 40% hiệu quả + 40% loss + 20% thời gian
        overall_improvement = 0.4 * efficiency_change + 0.4 * loss_change + 0.2 * time_change
        
        return {
            "efficiency_change": efficiency_change,
            "loss_change": loss_change,
            "time_change": time_change,
            "hyperparameter_changes": hyperparam_diff,
            "overall_improvement": overall_improvement,
            "is_better": overall_improvement > 0
        }
    
    def find_best_hyperparameters(self) -> Dict[str, Any]:
        """
        Tìm tập hyperparameters tốt nhất từ lịch sử huấn luyện.
        
        Returns:
            Dict[str, Any]: Thông tin về tập hyperparameters tốt nhất
        """
        if len(self.training_history["efficiency_scores"]) == 0:
            return {"message": "Không có dữ liệu huấn luyện"}
        
        # Tìm chỉ số của hiệu quả cao nhất
        best_idx = np.argmax(self.training_history["efficiency_scores"])
        
        # Lấy thông tin
        best_metrics = {
            "timestamp": self.training_history["timestamps"][best_idx],
            "training_time": self.training_history["training_time"][best_idx],
            "iterations": self.training_history["iterations"][best_idx],
            "final_loss": self.training_history["loss_values"][best_idx],
            "efficiency_score": self.training_history["efficiency_scores"][best_idx],
            "cpu_usage": self.training_history["cpu_usage"][best_idx],
            "memory_usage": self.training_history["memory_usage"][best_idx],
            "gpu_usage": self.training_history["gpu_usage"][best_idx],
            "hyperparameters": self.training_history["hyperparameters"][best_idx]
        }
        
        return {
            "best_hyperparameters": best_metrics["hyperparameters"],
            "metrics": best_metrics,
            "index": best_idx
        }
    
    def recommend_hyperparameters(self) -> Dict[str, Any]:
        """
        Đề xuất tập hyperparameters cho lần huấn luyện tiếp theo.
        
        Returns:
            Dict[str, Any]: Tập hyperparameters đề xuất và lý do
        """
        if len(self.training_history["efficiency_scores"]) < 2:
            return {"message": "Không đủ dữ liệu để đề xuất hyperparameters"}
        
        # Tìm tập hyperparameters tốt nhất
        best_result = self.find_best_hyperparameters()
        best_hyperparams = best_result["best_hyperparameters"]
        
        # Phân tích xu hướng
        trends = self._analyze_hyperparameter_trends()
        
        # Đề xuất tập hyperparameters mới dựa trên tập tốt nhất và xu hướng
        recommended_hyperparams = best_hyperparams.copy()
        adjustments = {}
        
        for param, trend in trends.items():
            if param in best_hyperparams:
                # Điều chỉnh dựa trên xu hướng
                if trend["correlation"] > 0.5:
                    # Tương quan dương mạnh - tăng giá trị
                    if isinstance(best_hyperparams[param], (int, float)):
                        new_value = best_hyperparams[param] * 1.1  # Tăng 10%
                        if param in trend["range"]:
                            # Đảm bảo nằm trong phạm vi hợp lệ
                            new_value = min(new_value, trend["range"][1])
                        recommended_hyperparams[param] = new_value
                        adjustments[param] = {"from": best_hyperparams[param], "to": new_value, "reason": "Tương quan dương mạnh"}
                
                elif trend["correlation"] < -0.5:
                    # Tương quan âm mạnh - giảm giá trị
                    if isinstance(best_hyperparams[param], (int, float)):
                        new_value = best_hyperparams[param] * 0.9  # Giảm 10%
                        if param in trend["range"]:
                            # Đảm bảo nằm trong phạm vi hợp lệ
                            new_value = max(new_value, trend["range"][0])
                        recommended_hyperparams[param] = new_value
                        adjustments[param] = {"from": best_hyperparams[param], "to": new_value, "reason": "Tương quan âm mạnh"}
        
        return {
            "best_hyperparameters": best_hyperparams,
            "recommended_hyperparameters": recommended_hyperparams,
            "adjustments": adjustments,
            "reasoning": "Đề xuất dựa trên tập hyperparameters tốt nhất và phân tích xu hướng",
            "trends": trends
        }
    
    def _analyze_hyperparameter_trends(self) -> Dict[str, Dict[str, Any]]:
        """
        Phân tích xu hướng của hyperparameters qua các lần huấn luyện.
        
        Returns:
            Dict[str, Dict[str, Any]]: Thông tin xu hướng cho mỗi hyperparameter
        """
        if len(self.training_history["hyperparameters"]) < 2:
            return {}
        
        # Tập hợp tất cả các hyperparameters đã sử dụng
        all_params = set()
        for hyperparams in self.training_history["hyperparameters"]:
            all_params.update(hyperparams.keys())
        
        trends = {}
        efficiency_scores = np.array(self.training_history["efficiency_scores"])
        
        for param in all_params:
            # Lấy giá trị của hyperparameter này qua các lần huấn luyện
            param_values = []
            for hyperparams in self.training_history["hyperparameters"]:
                if param in hyperparams and isinstance(hyperparams[param], (int, float)):
                    param_values.append(hyperparams[param])
                else:
                    param_values.append(None)
            
            # Loại bỏ các giá trị None
            valid_indices = [i for i, v in enumerate(param_values) if v is not None]
            valid_values = [param_values[i] for i in valid_indices]
            valid_scores = [efficiency_scores[i] for i in valid_indices]
            
            if len(valid_values) < 2:
                continue
            
            # Tính tương quan
            try:
                correlation = np.corrcoef(valid_values, valid_scores)[0, 1]
            except:
                correlation = 0.0
            
            # Ghi nhận xu hướng
            trends[param] = {
                "values": valid_values,
                "correlation": correlation,
                "range": (min(valid_values), max(valid_values)),
                "best_value": valid_values[np.argmax(valid_scores)],
                "importance": abs(correlation)
            }
        
        return trends
    
    def save_history(self, path: Union[str, Path]) -> None:
        """
        Lưu lịch sử hiệu quả vào file.
        
        Args:
            path (Union[str, Path]): Đường dẫn file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Chuyển đổi dữ liệu để lưu
        serializable_history = {
            "training_history": {
                "timestamps": [t.isoformat() for t in self.training_history["timestamps"]],
                "training_time": self.training_history["training_time"],
                "iterations": self.training_history["iterations"],
                "loss_values": self.training_history["loss_values"],
                "cpu_usage": self.training_history["cpu_usage"],
                "memory_usage": self.training_history["memory_usage"],
                "gpu_usage": self.training_history["gpu_usage"],
                "efficiency_scores": self.training_history["efficiency_scores"],
                "hyperparameters": self.training_history["hyperparameters"]
            },
            "baseline_metrics": self.baseline_metrics,
            "resource_thresholds": {
                "efficiency_threshold": self.efficiency_threshold,
                "resource_threshold": self.resource_threshold,
                "time_threshold": self.time_threshold
            },
            "last_updated": datetime.now().isoformat()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"Đã lưu lịch sử hiệu quả vào {path}")
    
    def load_history(self, path: Union[str, Path]) -> bool:
        """
        Tải lịch sử hiệu quả từ file.
        
        Args:
            path (Union[str, Path]): Đường dẫn file
            
        Returns:
            bool: True nếu tải thành công, False nếu không
        """
        path = Path(path)
        if not path.exists():
            self.logger.warning(f"Không tìm thấy file {path}")
            return False
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Khôi phục lịch sử huấn luyện
            self.training_history["timestamps"] = [datetime.fromisoformat(t) for t in data["training_history"]["timestamps"]]
            self.training_history["training_time"] = data["training_history"]["training_time"]
            self.training_history["iterations"] = data["training_history"]["iterations"]
            self.training_history["loss_values"] = data["training_history"]["loss_values"]
            self.training_history["cpu_usage"] = data["training_history"]["cpu_usage"]
            self.training_history["memory_usage"] = data["training_history"]["memory_usage"]
            self.training_history["gpu_usage"] = data["training_history"]["gpu_usage"]
            self.training_history["efficiency_scores"] = data["training_history"]["efficiency_scores"]
            self.training_history["hyperparameters"] = data["training_history"]["hyperparameters"]
            
            # Khôi phục baseline
            self.baseline_metrics = data.get("baseline_metrics")
            
            # Khôi phục ngưỡng
            if "resource_thresholds" in data:
                self.efficiency_threshold = data["resource_thresholds"].get("efficiency_threshold", self.efficiency_threshold)
                self.resource_threshold = data["resource_thresholds"].get("resource_threshold", self.resource_threshold)
                self.time_threshold = data["resource_thresholds"].get("time_threshold", self.time_threshold)
            
            self.logger.info(f"Đã tải lịch sử hiệu quả từ {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải lịch sử hiệu quả: {str(e)}")
            return False
    
    def plot_efficiency_metrics(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Vẽ biểu đồ các chỉ số hiệu quả.
        
        Args:
            figsize (Tuple[int, int], optional): Kích thước hình
            
        Returns:
            plt.Figure: Đối tượng Figure của matplotlib
        """
        fig, axs = plt.subplots(3, 2, figsize=figsize)
        
        # Chuyển đổi dữ liệu hiệu quả thành DataFrame
        data = pd.DataFrame({
            "timestamp": self.training_history["timestamps"],
            "efficiency_score": self.training_history["efficiency_scores"],
            "training_time": self.training_history["training_time"],
            "loss": self.training_history["loss_values"],
            "cpu_usage": self.training_history["cpu_usage"],
            "memory_usage": self.training_history["memory_usage"],
            "gpu_usage": self.training_history["gpu_usage"]
        })
        
        # 1. Hiệu quả theo thời gian
        ax1 = axs[0, 0]
        ax1.plot(data.index, data["efficiency_score"], 'b-', marker='o')
        ax1.set_title('Điểm hiệu quả theo thời gian')
        ax1.set_ylabel('Điểm hiệu quả')
        ax1.set_xlabel('Lần huấn luyện')
        ax1.grid(True, alpha=0.3)
        
        # Vẽ ngưỡng hiệu quả
        ax1.axhline(y=self.efficiency_threshold, color='r', linestyle='--', alpha=0.7)
        
        # 2. Thời gian huấn luyện
        ax2 = axs[0, 1]
        ax2.plot(data.index, data["training_time"], 'g-', marker='o')
        ax2.set_title('Thời gian huấn luyện')
        ax2.set_ylabel('Thời gian (giây)')
        ax2.set_xlabel('Lần huấn luyện')
        ax2.grid(True, alpha=0.3)
        
        # 3. Loss
        ax3 = axs[1, 0]
        ax3.plot(data.index, data["loss"], 'r-', marker='o')
        ax3.set_title('Giá trị Loss cuối cùng')
        ax3.set_ylabel('Loss')
        ax3.set_xlabel('Lần huấn luyện')
        ax3.grid(True, alpha=0.3)
        
        # 4. Sử dụng tài nguyên
        ax4 = axs[1, 1]
        ax4.plot(data.index, data["cpu_usage"], 'b-', marker='o', label='CPU')
        ax4.plot(data.index, data["memory_usage"], 'g-', marker='s', label='Memory')
        ax4.plot(data.index, data["gpu_usage"], 'r-', marker='^', label='GPU')
        ax4.set_title('Sử dụng tài nguyên')
        ax4.set_ylabel('Mức sử dụng (%)')
        ax4.set_xlabel('Lần huấn luyện')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Vẽ ngưỡng tài nguyên
        ax4.axhline(y=self.resource_threshold, color='r', linestyle='--', alpha=0.7)
        
        # 5. Hiệu quả vs Loss
        ax5 = axs[2, 0]
        ax5.scatter(data["loss"], data["efficiency_score"], c='purple')
        ax5.set_title('Hiệu quả vs Loss')
        ax5.set_xlabel('Loss')
        ax5.set_ylabel('Điểm hiệu quả')
        ax5.grid(True, alpha=0.3)
        
        # 6. Hiệu quả vs Thời gian
        ax6 = axs[2, 1]
        ax6.scatter(data["training_time"], data["efficiency_score"], c='orange')
        ax6.set_title('Hiệu quả vs Thời gian')
        ax6.set_xlabel('Thời gian huấn luyện (giây)')
        ax6.set_ylabel('Điểm hiệu quả')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig