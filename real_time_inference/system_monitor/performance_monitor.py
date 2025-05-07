"""
Giám sát hiệu suất hệ thống.
File này cung cấp các lớp và hàm để theo dõi hiệu suất của hệ thống giao dịch tự động,
bao gồm thời gian phản hồi, tỷ lệ thành công, hiệu suất giao dịch và các chỉ số khác.
"""

import os
import time
import logging
import threading
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
from collections import deque

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from logs.logger import SystemLogger

class PerformanceMetric:
    """
    Lớp đại diện cho một loại chỉ số hiệu suất.
    Lưu trữ giá trị và thống kê cho một chỉ số cụ thể.
    """
    
    def __init__(
        self, 
        name: str, 
        unit: str = "", 
        max_history_size: int = 1000,
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None,
        lower_is_better: bool = False
    ):
        """
        Khởi tạo chỉ số hiệu suất.
        
        Args:
            name: Tên chỉ số
            unit: Đơn vị đo (vd: "ms", "%", "MB/s")
            max_history_size: Số lượng giá trị tối đa lưu trữ trong lịch sử
            warning_threshold: Ngưỡng cảnh báo
            critical_threshold: Ngưỡng nghiêm trọng
            lower_is_better: True nếu giá trị thấp hơn là tốt hơn
        """
        self.name = name
        self.unit = unit
        self.max_history_size = max_history_size
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.lower_is_better = lower_is_better
        
        # Lưu trữ giá trị
        self.values = deque(maxlen=max_history_size)
        self.timestamps = deque(maxlen=max_history_size)
        
        # Thống kê
        self.min_value = None
        self.max_value = None
        self.sum_value = 0
        self.count = 0
        
        # Giá trị hiện tại và trước đó
        self.current_value = None
        self.previous_value = None
    
    def add_value(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """
        Thêm giá trị mới cho chỉ số.
        
        Args:
            value: Giá trị mới
            timestamp: Thời gian ghi nhận (None để sử dụng thời gian hiện tại)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Cập nhật giá trị trước đó
        self.previous_value = self.current_value
        self.current_value = value
        
        # Thêm vào lịch sử
        self.values.append(value)
        self.timestamps.append(timestamp)
        
        # Cập nhật thống kê
        if self.min_value is None or value < self.min_value:
            self.min_value = value
        
        if self.max_value is None or value > self.max_value:
            self.max_value = value
        
        self.sum_value += value
        self.count += 1
    
    def get_mean(self) -> Optional[float]:
        """
        Tính giá trị trung bình.
        
        Returns:
            Giá trị trung bình hoặc None nếu không có dữ liệu
        """
        if self.count == 0:
            return None
        return self.sum_value / self.count
    
    def get_recent_mean(self, n: int = 10) -> Optional[float]:
        """
        Tính giá trị trung bình của n giá trị gần nhất.
        
        Args:
            n: Số lượng giá trị gần nhất cần tính
            
        Returns:
            Giá trị trung bình hoặc None nếu không có đủ dữ liệu
        """
        if len(self.values) == 0:
            return None
        
        n = min(n, len(self.values))
        recent_values = list(self.values)[-n:]
        return sum(recent_values) / len(recent_values)
    
    def get_percentile(self, p: float) -> Optional[float]:
        """
        Tính giá trị phân vị thứ p.
        
        Args:
            p: Phân vị (0-100)
            
        Returns:
            Giá trị tại phân vị p hoặc None nếu không có dữ liệu
        """
        if len(self.values) == 0:
            return None
        
        return np.percentile(list(self.values), p)
    
    def get_values_as_array(self) -> np.ndarray:
        """
        Lấy mảng các giá trị.
        
        Returns:
            Mảng NumPy chứa các giá trị
        """
        return np.array(self.values)
    
    def get_timestamps_as_array(self) -> np.ndarray:
        """
        Lấy mảng các thời gian.
        
        Returns:
            Mảng NumPy chứa các thời gian
        """
        return np.array(self.timestamps)
    
    def get_status(self) -> str:
        """
        Lấy trạng thái hiện tại của chỉ số dựa trên ngưỡng.
        
        Returns:
            "normal", "warning", "critical" hoặc "unknown"
        """
        if self.current_value is None:
            return "unknown"
        
        if self.critical_threshold is not None:
            if (self.lower_is_better and self.current_value >= self.critical_threshold) or \
               (not self.lower_is_better and self.current_value <= self.critical_threshold):
                return "critical"
        
        if self.warning_threshold is not None:
            if (self.lower_is_better and self.current_value >= self.warning_threshold) or \
               (not self.lower_is_better and self.current_value <= self.warning_threshold):
                return "warning"
        
        return "normal"
    
    def get_change_percent(self) -> Optional[float]:
        """
        Tính phần trăm thay đổi so với giá trị trước đó.
        
        Returns:
            Phần trăm thay đổi hoặc None nếu không có đủ dữ liệu
        """
        if self.current_value is None or self.previous_value is None or self.previous_value == 0:
            return None
        
        return ((self.current_value - self.previous_value) / self.previous_value) * 100
    
    def reset_stats(self) -> None:
        """
        Đặt lại thống kê.
        """
        self.values.clear()
        self.timestamps.clear()
        self.min_value = None
        self.max_value = None
        self.sum_value = 0
        self.count = 0
        self.current_value = None
        self.previous_value = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thành dictionary.
        
        Returns:
            Dict chứa thông tin chỉ số
        """
        return {
            "name": self.name,
            "unit": self.unit,
            "current": self.current_value,
            "min": self.min_value,
            "max": self.max_value,
            "mean": self.get_mean(),
            "count": self.count,
            "recent_mean": self.get_recent_mean(),
            "status": self.get_status(),
            "change_percent": self.get_change_percent()
        }

class PerformanceMonitor:
    """
    Lớp giám sát hiệu suất hệ thống.
    Thu thập và phân tích các chỉ số hiệu suất từ nhiều nguồn khác nhau.
    """
    
    def __init__(
        self,
        check_interval: float = 60.0,
        history_size: int = 1000,
        log_interval: int = 5,
        storage_path: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo giám sát hiệu suất.
        
        Args:
            check_interval: Khoảng thời gian giữa các lần kiểm tra (giây)
            history_size: Số lượng giá trị tối đa lưu trữ trong lịch sử
            log_interval: Số lần kiểm tra trước khi ghi log (5 = ghi log mỗi 5 lần kiểm tra)
            storage_path: Đường dẫn lưu trữ dữ liệu hiệu suất
            logger: Logger tùy chỉnh
        """
        # Logger
        self.logger = logger or SystemLogger("performance_monitor")
        
        # Cấu hình
        self.system_config = get_system_config()
        self.check_interval = check_interval
        self.history_size = history_size
        self.log_interval = log_interval
        
        # Đường dẫn lưu trữ
        if storage_path is None:
            storage_path = Path("logs/performance")
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo các chỉ số
        self.metrics = {}
        self._init_default_metrics()
        
        # Callbacks khi có vấn đề về hiệu suất
        self.performance_callbacks = []
        
        # Thread giám sát
        self.monitor_thread = None
        self.stop_flag = threading.Event()
        self.check_count = 0
        
        # Thời gian bắt đầu
        self.start_time = datetime.now()
        
        self.logger.info(f"Khởi tạo PerformanceMonitor với chu kỳ kiểm tra {check_interval} giây")
    
    def _init_default_metrics(self) -> None:
        """
        Khởi tạo các chỉ số mặc định.
        """
        # Chỉ số hệ thống
        self.add_metric("cpu_usage", unit="%", warning_threshold=80, critical_threshold=95, lower_is_better=True)
        self.add_metric("memory_usage", unit="%", warning_threshold=80, critical_threshold=95, lower_is_better=True)
        self.add_metric("disk_usage", unit="%", warning_threshold=85, critical_threshold=95, lower_is_better=True)
        
        # Chỉ số mạng
        self.add_metric("network_latency", unit="ms", warning_threshold=200, critical_threshold=500, lower_is_better=True)
        self.add_metric("api_response_time", unit="ms", warning_threshold=1000, critical_threshold=5000, lower_is_better=True)
        
        # Chỉ số giao dịch
        self.add_metric("order_success_rate", unit="%", warning_threshold=90, critical_threshold=80)
        self.add_metric("order_execution_time", unit="ms", warning_threshold=500, critical_threshold=2000, lower_is_better=True)
        
        # Chỉ số tài chính
        self.add_metric("daily_pnl", unit="$")
        self.add_metric("drawdown", unit="%", warning_threshold=10, critical_threshold=20, lower_is_better=True)
        
        # Chỉ số dự đoán
        self.add_metric("prediction_accuracy", unit="%", warning_threshold=65, critical_threshold=50)
        self.add_metric("model_inference_time", unit="ms", warning_threshold=100, critical_threshold=500, lower_is_better=True)
    
    def add_metric(
        self, 
        name: str, 
        unit: str = "", 
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None,
        lower_is_better: bool = False
    ) -> PerformanceMetric:
        """
        Thêm chỉ số mới.
        
        Args:
            name: Tên chỉ số
            unit: Đơn vị đo
            warning_threshold: Ngưỡng cảnh báo
            critical_threshold: Ngưỡng nghiêm trọng
            lower_is_better: True nếu giá trị thấp hơn là tốt hơn
            
        Returns:
            Đối tượng PerformanceMetric mới tạo
        """
        if name in self.metrics:
            self.logger.warning(f"Chỉ số '{name}' đã tồn tại và sẽ bị ghi đè")
        
        metric = PerformanceMetric(
            name=name,
            unit=unit,
            max_history_size=self.history_size,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            lower_is_better=lower_is_better
        )
        
        self.metrics[name] = metric
        return metric
    
    def remove_metric(self, name: str) -> bool:
        """
        Xóa chỉ số.
        
        Args:
            name: Tên chỉ số cần xóa
            
        Returns:
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if name in self.metrics:
            del self.metrics[name]
            self.logger.debug(f"Đã xóa chỉ số '{name}'")
            return True
        
        self.logger.warning(f"Không tìm thấy chỉ số '{name}' để xóa")
        return False
    
    def update_metric(self, name: str, value: float, timestamp: Optional[datetime] = None) -> bool:
        """
        Cập nhật giá trị cho một chỉ số.
        
        Args:
            name: Tên chỉ số
            value: Giá trị mới
            timestamp: Thời gian ghi nhận
            
        Returns:
            True nếu cập nhật thành công, False nếu không tìm thấy chỉ số
        """
        if name not in self.metrics:
            self.logger.warning(f"Không tìm thấy chỉ số '{name}' để cập nhật")
            return False
        
        self.metrics[name].add_value(value, timestamp)
        
        # Kiểm tra trạng thái sau khi cập nhật
        status = self.metrics[name].get_status()
        if status in ["warning", "critical"]:
            message = f"Chỉ số '{name}' ở mức {status}: {value} {self.metrics[name].unit}"
            
            if status == "warning":
                self.logger.warning(message)
            else:
                self.logger.error(message)
            
            # Gọi callbacks nếu có
            if self.performance_callbacks:
                metric_info = self.metrics[name].to_dict()
                for callback in self.performance_callbacks:
                    try:
                        callback(name, status, metric_info)
                    except Exception as e:
                        self.logger.error(f"Lỗi khi gọi callback hiệu suất: {str(e)}", exc_info=True)
        
        return True
    
    def get_metric(self, name: str) -> Optional[PerformanceMetric]:
        """
        Lấy đối tượng chỉ số theo tên.
        
        Args:
            name: Tên chỉ số
            
        Returns:
            Đối tượng PerformanceMetric hoặc None nếu không tìm thấy
        """
        return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Lấy tất cả các chỉ số dưới dạng dictionary.
        
        Returns:
            Dict chứa thông tin tất cả các chỉ số
        """
        return {name: metric.to_dict() for name, metric in self.metrics.items()}
    
    def start_monitoring(self) -> bool:
        """
        Bắt đầu giám sát hiệu suất trong thread riêng.
        
        Returns:
            True nếu bắt đầu thành công, False nếu không
        """
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            self.logger.warning("Giám sát hiệu suất đã đang chạy")
            return False
        
        # Đặt lại cờ dừng
        self.stop_flag.clear()
        
        # Tạo và khởi động thread mới
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceMonitorThread",
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("Đã bắt đầu giám sát hiệu suất hệ thống")
        return True
    
    def stop_monitoring(self) -> bool:
        """
        Dừng giám sát hiệu suất.
        
        Returns:
            True nếu dừng thành công, False nếu không
        """
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.logger.warning("Giám sát hiệu suất không đang chạy")
            return False
        
        # Đặt cờ dừng
        self.stop_flag.set()
        
        # Chờ thread kết thúc
        self.monitor_thread.join(timeout=5.0)
        
        if self.monitor_thread.is_alive():
            self.logger.warning("Không thể dừng thread giám sát hiệu suất")
            return False
        
        self.monitor_thread = None
        self.logger.info("Đã dừng giám sát hiệu suất hệ thống")
        return True
    
    def _monitoring_loop(self) -> None:
        """
        Vòng lặp giám sát chính.
        """
        self.logger.info("Vòng lặp giám sát hiệu suất bắt đầu")
        
        while not self.stop_flag.is_set():
            try:
                # Thực hiện kiểm tra hiệu suất
                self.check_performance()
                
                # Tăng bộ đếm kiểm tra
                self.check_count += 1
                
                # Ghi log theo khoảng thời gian
                if self.check_count % self.log_interval == 0:
                    self._log_performance_stats()
                
                # Lưu dữ liệu theo khoảng thời gian
                if self.check_count % (self.log_interval * 10) == 0:
                    self.save_performance_data()
                
                # Đợi đến lần kiểm tra tiếp theo
                self.stop_flag.wait(self.check_interval)
            except Exception as e:
                self.logger.error(f"Lỗi trong vòng lặp giám sát: {str(e)}", exc_info=True)
                # Đợi một chút trước khi thử lại
                time.sleep(5)
        
        self.logger.info("Vòng lặp giám sát hiệu suất đã kết thúc")
    
    def check_performance(self) -> Dict[str, Any]:
        """
        Thực hiện kiểm tra hiệu suất hệ thống.
        Phương thức này nên được ghi đè trong lớp con để thực hiện
        các kiểm tra cụ thể tùy theo nhu cầu.
        
        Returns:
            Dict chứa kết quả kiểm tra
        """
        # Lưu ý: Đây chỉ là phương thức ảo cơ bản
        # Trong triển khai thực tế, nên ghi đè phương thức này
        # để thực hiện các kiểm tra cụ thể
        
        # Tạo kết quả trống
        results = {}
        
        # Gọi các phương thức kiểm tra cụ thể
        try:
            self._check_system_metrics()
            self._check_network_metrics()
            self._check_trading_metrics()
            self._check_financial_metrics()
            self._check_prediction_metrics()
        except Exception as e:
            self.logger.error(f"Lỗi khi kiểm tra hiệu suất: {str(e)}", exc_info=True)
        
        # Thu thập kết quả từ tất cả các chỉ số
        for name, metric in self.metrics.items():
            results[name] = metric.to_dict()
        
        return results
    
    def _check_system_metrics(self) -> None:
        """
        Kiểm tra các chỉ số hệ thống (CPU, memory, disk).
        """
        # Giả lập việc cập nhật, trong triển khai thực tế sẽ lấy dữ liệu thực
        import psutil
        
        # CPU
        cpu_usage = psutil.cpu_percent(interval=0.5)
        self.update_metric("cpu_usage", cpu_usage)
        
        # Memory
        memory_usage = psutil.virtual_memory().percent
        self.update_metric("memory_usage", memory_usage)
        
        # Disk
        disk_usage = psutil.disk_usage(Path.cwd()).percent
        self.update_metric("disk_usage", disk_usage)
    
    def _check_network_metrics(self) -> None:
        """
        Kiểm tra các chỉ số mạng (độ trễ, thời gian phản hồi API).
        """
        # Giả lập việc kiểm tra, trong triển khai thực tế sẽ thực hiện kiểm tra thực
        
        # Độ trễ mạng (giả lập)
        import random
        network_latency = random.uniform(10, 300)  # ms
        self.update_metric("network_latency", network_latency)
        
        # Thời gian phản hồi API (giả lập)
        api_response_time = random.uniform(50, 2000)  # ms
        self.update_metric("api_response_time", api_response_time)
    
    def _check_trading_metrics(self) -> None:
        """
        Kiểm tra các chỉ số giao dịch.
        """
        # Trong triển khai thực tế, sẽ lấy dữ liệu từ hệ thống giao dịch
        pass
    
    def _check_financial_metrics(self) -> None:
        """
        Kiểm tra các chỉ số tài chính.
        """
        # Trong triển khai thực tế, sẽ lấy dữ liệu từ hệ thống quản lý tài chính
        pass
    
    def _check_prediction_metrics(self) -> None:
        """
        Kiểm tra các chỉ số dự đoán.
        """
        # Trong triển khai thực tế, sẽ lấy dữ liệu từ mô hình dự đoán
        pass
    
    def _log_performance_stats(self) -> None:
        """
        Ghi log thống kê hiệu suất.
        """
        # Lấy các chỉ số quan trọng
        critical_metrics = []
        warning_metrics = []
        
        for name, metric in self.metrics.items():
            status = metric.get_status()
            
            if status == "critical":
                critical_metrics.append(f"{name}: {metric.current_value}{metric.unit}")
            elif status == "warning":
                warning_metrics.append(f"{name}: {metric.current_value}{metric.unit}")
        
        # Ghi log với mức độ phù hợp
        if critical_metrics:
            self.logger.error(f"Chỉ số nghiêm trọng: {', '.join(critical_metrics)}")
        elif warning_metrics:
            self.logger.warning(f"Chỉ số cảnh báo: {', '.join(warning_metrics)}")
        else:
            # Chỉ log các chỉ số quan trọng
            system_metrics = []
            
            for name in ["cpu_usage", "memory_usage", "order_success_rate"]:
                metric = self.metrics.get(name)
                if metric and metric.current_value is not None:
                    system_metrics.append(f"{name}: {metric.current_value}{metric.unit}")
            
            if system_metrics:
                self.logger.info(f"Hiệu suất hệ thống: {', '.join(system_metrics)}")
    
    def register_callback(self, callback: Callable[[str, str, Dict[str, Any]], None]) -> None:
        """
        Đăng ký callback để nhận thông báo khi có vấn đề hiệu suất.
        
        Args:
            callback: Hàm callback nhận (metric_name, status, metric_info)
        """
        if callback not in self.performance_callbacks:
            self.performance_callbacks.append(callback)
            self.logger.debug(f"Đã đăng ký callback hiệu suất: {callback.__name__}")
    
    def unregister_callback(self, callback: Callable[[str, str, Dict[str, Any]], None]) -> bool:
        """
        Hủy đăng ký callback.
        
        Args:
            callback: Hàm callback cần hủy
            
        Returns:
            True nếu hủy thành công, False nếu không tìm thấy
        """
        if callback in self.performance_callbacks:
            self.performance_callbacks.remove(callback)
            self.logger.debug(f"Đã hủy đăng ký callback hiệu suất: {callback.__name__}")
            return True
        return False
    
    def save_performance_data(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Lưu dữ liệu hiệu suất hiện tại vào file.
        
        Args:
            file_path: Đường dẫn file (None để sử dụng đường dẫn mặc định)
            
        Returns:
            True nếu lưu thành công, False nếu không
        """
        if file_path is None:
            # Tạo tên file dựa trên thời gian
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.storage_path / f"performance_{timestamp}.json"
        
        try:
            # Đảm bảo thư mục tồn tại
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Thu thập dữ liệu hiệu suất
            performance_data = {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "metrics": self.get_all_metrics()
            }
            
            # Lưu vào file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(performance_data, f, ensure_ascii=False, indent=4)
            
            self.logger.info(f"Đã lưu dữ liệu hiệu suất vào {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu dữ liệu hiệu suất: {str(e)}", exc_info=True)
            return False
    
    def load_performance_data(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Tải dữ liệu hiệu suất từ file.
        
        Args:
            file_path: Đường dẫn file
            
        Returns:
            Dict dữ liệu hiệu suất hoặc None nếu không thành công
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                performance_data = json.load(f)
            
            self.logger.info(f"Đã tải dữ liệu hiệu suất từ {file_path}")
            return performance_data
        except Exception as e:
            self.logger.error(f"Lỗi khi tải dữ liệu hiệu suất: {str(e)}", exc_info=True)
            return None
    
    def get_performance_history(
        self, 
        metric_name: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Lấy lịch sử hiệu suất của một chỉ số trong khoảng thời gian.
        
        Args:
            metric_name: Tên chỉ số
            start_time: Thời gian bắt đầu
            end_time: Thời gian kết thúc
            
        Returns:
            DataFrame chứa lịch sử hiệu suất hoặc None nếu không tìm thấy
        """
        metric = self.get_metric(metric_name)
        
        if metric is None:
            self.logger.warning(f"Không tìm thấy chỉ số '{metric_name}'")
            return None
        
        if len(metric.values) == 0:
            return None
        
        # Chuyển đổi thành DataFrame
        df = pd.DataFrame({
            'timestamp': metric.timestamps,
            'value': metric.values
        })
        
        # Lọc theo khoảng thời gian
        if start_time is not None:
            df = df[df['timestamp'] >= start_time]
        
        if end_time is not None:
            df = df[df['timestamp'] <= end_time]
        
        return df
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Lấy bản tóm tắt hiệu suất hệ thống.
        
        Returns:
            Dict chứa tóm tắt hiệu suất
        """
        # Tạo Dict tóm tắt
        summary = {
            "timestamp": datetime.now().isoformat(),
            "uptime": str(datetime.now() - self.start_time),
            "system_status": self._get_overall_status(),
            "metrics": {}
        }
        
        # Thêm thông tin tóm tắt cho từng chỉ số
        for name, metric in self.metrics.items():
            if metric.current_value is not None:
                summary["metrics"][name] = {
                    "value": metric.current_value,
                    "unit": metric.unit,
                    "status": metric.get_status()
                }
        
        return summary
    
    def _get_overall_status(self) -> str:
        """
        Xác định trạng thái tổng thể dựa trên tất cả các chỉ số.
        
        Returns:
            "normal", "caution", "warning", hoặc "critical"
        """
        has_critical = False
        warning_count = 0
        
        for metric in self.metrics.values():
            status = metric.get_status()
            
            if status == "critical":
                has_critical = True
            elif status == "warning":
                warning_count += 1
        
        if has_critical:
            return "critical"
        elif warning_count > 2:
            return "warning"
        elif warning_count > 0:
            return "caution"
        else:
            return "normal"
    
    def get_health_score(self) -> int:
        """
        Tính điểm sức khỏe hệ thống (0-100).
        
        Returns:
            Điểm sức khỏe (0-100)
        """
        # Khởi tạo điểm tổng thể
        base_score = 100
        penalty = 0
        active_metrics = 0
        
        # Tính điểm phạt dựa trên các chỉ số
        for name, metric in self.metrics.items():
            if metric.current_value is None:
                continue
            
            active_metrics += 1
            status = metric.get_status()
            
            if status == "critical":
                # Trừ 15-20 điểm cho mỗi chỉ số nghiêm trọng
                penalty += 20
            elif status == "warning":
                # Trừ 5-10 điểm cho mỗi chỉ số cảnh báo
                penalty += 10
        
        # Tính điểm cuối cùng
        final_score = max(0, base_score - penalty)
        
        # Nếu không có chỉ số nào hoạt động, trả về 0
        if active_metrics == 0:
            return 0
        
        return final_score
    
    def reset_all_metrics(self) -> None:
        """
        Đặt lại tất cả các chỉ số.
        """
        for metric in self.metrics.values():
            metric.reset_stats()
        
        self.logger.info("Đã đặt lại tất cả các chỉ số hiệu suất")

# Singleton instance
_performance_monitor_instance = None

def get_performance_monitor() -> PerformanceMonitor:
    """
    Lấy instance singleton của PerformanceMonitor.
    
    Returns:
        Instance PerformanceMonitor
    """
    global _performance_monitor_instance
    if _performance_monitor_instance is None:
        _performance_monitor_instance = PerformanceMonitor()
    return _performance_monitor_instance