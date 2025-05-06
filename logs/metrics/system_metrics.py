"""
Module đo lường hệ thống.
File này cung cấp các lớp và hàm để thu thập, tính toán và ghi log
các chỉ số hiệu suất của hệ thống giao dịch tự động, bao gồm
CPU, bộ nhớ, độ trễ mạng, thời gian phản hồi API, v.v.
"""

import os
import time
import psutil
import socket
import platform
import threading
import json
import logging
import statistics
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd

# Import các module từ hệ thống
from config.constants import MAX_LATENCY_MS, API_TIMEOUT
from config.system_config import get_system_config
from logs.logger import get_system_logger, SystemLogger

class SystemMetric:
    """
    Lớp cơ sở cho các chỉ số hệ thống.
    Định nghĩa giao diện chung cho tất cả các loại chỉ số.
    """
    
    def __init__(self, name: str, description: str, unit: str = ""):
        """
        Khởi tạo một chỉ số hệ thống.
        
        Args:
            name: Tên của chỉ số
            description: Mô tả của chỉ số
            unit: Đơn vị đo lường (nếu có)
        """
        self.name = name
        self.description = description
        self.unit = unit
        self.values: List[float] = []
        self.timestamps: List[datetime] = []
    
    def add_value(self, value: float) -> None:
        """
        Thêm một giá trị mới cho chỉ số.
        
        Args:
            value: Giá trị cần thêm
        """
        self.values.append(value)
        self.timestamps.append(datetime.now())
    
    def get_latest(self) -> Tuple[float, datetime]:
        """
        Lấy giá trị mới nhất của chỉ số.
        
        Returns:
            Tuple (giá trị, thời gian) mới nhất
        """
        if not self.values:
            return 0.0, datetime.now()
        return self.values[-1], self.timestamps[-1]
    
    def get_average(self, window: int = None) -> float:
        """
        Tính giá trị trung bình của chỉ số.
        
        Args:
            window: Số lượng giá trị gần nhất để tính trung bình (None = tất cả)
            
        Returns:
            Giá trị trung bình
        """
        if not self.values:
            return 0.0
        
        if window is not None and window > 0:
            values_to_avg = self.values[-min(window, len(self.values)):]
        else:
            values_to_avg = self.values
        
        return sum(values_to_avg) / len(values_to_avg)
    
    def get_min(self, window: int = None) -> float:
        """
        Tìm giá trị nhỏ nhất của chỉ số.
        
        Args:
            window: Số lượng giá trị gần nhất để tìm (None = tất cả)
            
        Returns:
            Giá trị nhỏ nhất
        """
        if not self.values:
            return 0.0
        
        if window is not None and window > 0:
            values_to_check = self.values[-min(window, len(self.values)):]
        else:
            values_to_check = self.values
        
        return min(values_to_check)
    
    def get_max(self, window: int = None) -> float:
        """
        Tìm giá trị lớn nhất của chỉ số.
        
        Args:
            window: Số lượng giá trị gần nhất để tìm (None = tất cả)
            
        Returns:
            Giá trị lớn nhất
        """
        if not self.values:
            return 0.0
        
        if window is not None and window > 0:
            values_to_check = self.values[-min(window, len(self.values)):]
        else:
            values_to_check = self.values
        
        return max(values_to_check)
    
    def get_std_dev(self, window: int = None) -> float:
        """
        Tính độ lệch chuẩn của chỉ số.
        
        Args:
            window: Số lượng giá trị gần nhất để tính (None = tất cả)
            
        Returns:
            Độ lệch chuẩn
        """
        if not self.values or len(self.values) < 2:
            return 0.0
        
        if window is not None and window > 0:
            values_to_use = self.values[-min(window, len(self.values)):]
        else:
            values_to_use = self.values
        
        return statistics.stdev(values_to_use)
    
    def reset(self) -> None:
        """Xóa tất cả các giá trị đã lưu."""
        self.values = []
        self.timestamps = []
    
    def get_stats(self, window: int = None) -> Dict[str, Any]:
        """
        Lấy thống kê tổng hợp về chỉ số.
        
        Args:
            window: Số lượng giá trị gần nhất để tính (None = tất cả)
            
        Returns:
            Dictionary chứa các thống kê
        """
        latest_value, latest_time = self.get_latest()
        
        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "latest_value": latest_value,
            "latest_timestamp": latest_time.isoformat(),
            "average": self.get_average(window),
            "min": self.get_min(window),
            "max": self.get_max(window),
            "std_dev": self.get_std_dev(window),
            "sample_count": len(self.values) if window is None else min(window, len(self.values))
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Chuyển đổi dữ liệu chỉ số sang DataFrame.
        
        Returns:
            DataFrame chứa dữ liệu chỉ số
        """
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'value': self.values
        })

class CPUMetric(SystemMetric):
    """Chỉ số về sử dụng CPU."""
    
    def __init__(self):
        """Khởi tạo chỉ số CPU."""
        super().__init__(
            name="cpu_usage",
            description="Phần trăm sử dụng CPU",
            unit="%"
        )
    
    def measure(self) -> float:
        """
        Đo và ghi lại phần trăm sử dụng CPU.
        
        Returns:
            Giá trị đo được
        """
        # Đo trong khoảng 0.1 giây để có kết quả chính xác hơn
        cpu_percent = psutil.cpu_usage(interval=0.1)
        self.add_value(cpu_percent)
        return cpu_percent
    
    def get_per_core_usage(self) -> List[float]:
        """
        Đo phần trăm sử dụng cho từng lõi CPU.
        
        Returns:
            Danh sách phần trăm sử dụng của từng lõi
        """
        per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        return per_core
    
    def get_load_average(self) -> List[float]:
        """
        Lấy thông tin về load trung bình của hệ thống.
        
        Returns:
            Danh sách load trung bình [1 phút, 5 phút, 15 phút]
        """
        try:
            return psutil.getloadavg()
        except (AttributeError, OSError):
            # Trên các hệ điều hành không hỗ trợ
            return [0.0, 0.0, 0.0]

class MemoryMetric(SystemMetric):
    """Chỉ số về sử dụng bộ nhớ."""
    
    def __init__(self):
        """Khởi tạo chỉ số bộ nhớ."""
        super().__init__(
            name="memory_usage",
            description="Phần trăm sử dụng bộ nhớ RAM",
            unit="%"
        )
        self.system_memory = None
    
    def measure(self) -> float:
        """
        Đo và ghi lại phần trăm sử dụng bộ nhớ RAM.
        
        Returns:
            Giá trị đo được
        """
        memory = psutil.virtual_memory()
        self.system_memory = memory
        memory_percent = memory.percent
        self.add_value(memory_percent)
        return memory_percent
    
    def get_memory_details(self) -> Dict[str, Any]:
        """
        Lấy thông tin chi tiết về sử dụng bộ nhớ.
        
        Returns:
            Dictionary chứa thông tin chi tiết về bộ nhớ
        """
        if self.system_memory is None:
            self.measure()
        
        memory = self.system_memory
        return {
            "total": memory.total / (1024 * 1024 * 1024),  # GB
            "available": memory.available / (1024 * 1024 * 1024),  # GB
            "used": memory.used / (1024 * 1024 * 1024),  # GB
            "percent": memory.percent
        }
    
    def measure_process_memory(self, pid: int = None) -> Dict[str, float]:
        """
        Đo bộ nhớ được sử dụng bởi một quy trình cụ thể.
        
        Args:
            pid: ID của quy trình (mặc định là quy trình hiện tại)
            
        Returns:
            Dictionary chứa thông tin bộ nhớ của quy trình
        """
        if pid is None:
            pid = os.getpid()
        
        try:
            process = psutil.Process(pid)
            memory_info = process.memory_info()
            
            return {
                "rss": memory_info.rss / (1024 * 1024),  # MB
                "vms": memory_info.vms / (1024 * 1024),  # MB
                "percent": process.memory_percent()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {
                "rss": 0.0,
                "vms": 0.0,
                "percent": 0.0
            }

class DiskMetric(SystemMetric):
    """Chỉ số về sử dụng và hiệu suất đĩa."""
    
    def __init__(self, path: str = "/"):
        """
        Khởi tạo chỉ số đĩa.
        
        Args:
            path: Đường dẫn thư mục để kiểm tra (mặc định là thư mục gốc)
        """
        super().__init__(
            name="disk_usage",
            description=f"Phần trăm sử dụng đĩa tại {path}",
            unit="%"
        )
        self.path = path
    
    def measure(self) -> float:
        """
        Đo và ghi lại phần trăm sử dụng đĩa.
        
        Returns:
            Giá trị đo được
        """
        usage = psutil.disk_usage(self.path)
        disk_percent = usage.percent
        self.add_value(disk_percent)
        return disk_percent
    
    def get_disk_details(self) -> Dict[str, Any]:
        """
        Lấy thông tin chi tiết về sử dụng đĩa.
        
        Returns:
            Dictionary chứa thông tin chi tiết về đĩa
        """
        usage = psutil.disk_usage(self.path)
        return {
            "total": usage.total / (1024 * 1024 * 1024),  # GB
            "used": usage.used / (1024 * 1024 * 1024),  # GB
            "free": usage.free / (1024 * 1024 * 1024),  # GB
            "percent": usage.percent
        }
    
    def get_io_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê về I/O đĩa.
        
        Returns:
            Dictionary chứa thông kê I/O
        """
        try:
            io_counters = psutil.disk_io_counters()
            return {
                "read_count": io_counters.read_count,
                "write_count": io_counters.write_count,
                "read_bytes": io_counters.read_bytes / (1024 * 1024),  # MB
                "write_bytes": io_counters.write_bytes / (1024 * 1024),  # MB
                "read_time": io_counters.read_time,  # ms
                "write_time": io_counters.write_time  # ms
            }
        except (AttributeError, OSError):
            return {
                "read_count": 0,
                "write_count": 0,
                "read_bytes": 0.0,
                "write_bytes": 0.0,
                "read_time": 0,
                "write_time": 0
            }

class NetworkMetric(SystemMetric):
    """Chỉ số về hiệu suất mạng."""
    
    def __init__(self):
        """Khởi tạo chỉ số mạng."""
        super().__init__(
            name="network_latency",
            description="Độ trễ mạng tới các máy chủ quan trọng",
            unit="ms"
        )
        self.server_latencies = {}
        self.bandwidth_usage = {"sent": [], "received": [], "timestamps": []}
        self.last_io = None
    
    def measure_latency(self, host: str = "8.8.8.8", timeout: float = 2.0, count: int = 3) -> float:
        """
        Đo độ trễ tới một máy chủ cụ thể bằng cách ping.
        
        Args:
            host: Địa chỉ máy chủ để ping
            timeout: Thời gian chờ tối đa (giây)
            count: Số lần ping
            
        Returns:
            Độ trễ trung bình (ms)
        """
        import subprocess
        
        try:
            if platform.system().lower() == "windows":
                ping_param = "-n"
            else:
                ping_param = "-c"
            
            cmd = ["ping", ping_param, str(count), "-W", str(int(timeout * 1000)), host]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Phân tích kết quả để lấy thời gian
            if result.returncode == 0:
                output = result.stdout
                if "time=" in output or "thời gian=" in output:
                    # Trích xuất các giá trị thời gian
                    times = []
                    for line in output.split("\n"):
                        if "time=" in line:
                            try:
                                time_str = line.split("time=")[1].split()[0]
                                times.append(float(time_str.strip("ms")))
                            except (IndexError, ValueError):
                                pass
                        elif "thời gian=" in line:
                            try:
                                time_str = line.split("thời gian=")[1].split()[0]
                                times.append(float(time_str.strip("ms")))
                            except (IndexError, ValueError):
                                pass
                    
                    if times:
                        avg_latency = sum(times) / len(times)
                        self.server_latencies[host] = avg_latency
                        return avg_latency
            
            # Nếu không thể đo được
            self.server_latencies[host] = MAX_LATENCY_MS
            return MAX_LATENCY_MS
            
        except Exception:
            self.server_latencies[host] = MAX_LATENCY_MS
            return MAX_LATENCY_MS
    
    def measure_socket_latency(self, host: str, port: int, timeout: float = 2.0) -> float:
        """
        Đo độ trễ kết nối socket tới một dịch vụ cụ thể.
        
        Args:
            host: Địa chỉ máy chủ
            port: Cổng dịch vụ
            timeout: Thời gian chờ tối đa (giây)
            
        Returns:
            Độ trễ (ms)
        """
        try:
            start_time = time.time()
            
            # Tạo socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            # Kết nối
            sock.connect((host, port))
            
            # Tính thời gian
            latency = (time.time() - start_time) * 1000  # Chuyển sang ms
            sock.close()
            
            # Lưu kết quả
            self.server_latencies[f"{host}:{port}"] = latency
            return latency
            
        except (socket.timeout, socket.error):
            # Nếu không thể kết nối
            self.server_latencies[f"{host}:{port}"] = MAX_LATENCY_MS
            return MAX_LATENCY_MS
    
    def measure_bandwidth(self) -> Dict[str, float]:
        """
        Đo lượng băng thông mạng đang được sử dụng.
        
        Returns:
            Dictionary chứa thông tin băng thông gửi/nhận (MB/s)
        """
        current_io = psutil.net_io_counters()
        current_time = time.time()
        
        # Nếu đây là lần đo đầu tiên
        if self.last_io is None:
            self.last_io = (current_io, current_time)
            return {"sent": 0.0, "received": 0.0}
        
        # Tính toán băng thông
        last_io, last_time = self.last_io
        time_delta = current_time - last_time
        
        # Tránh chia cho 0
        if time_delta < 0.001:
            time_delta = 0.001
        
        # Tính toán MB/s
        sent = (current_io.bytes_sent - last_io.bytes_sent) / (1024 * 1024 * time_delta)
        received = (current_io.bytes_recv - last_io.bytes_recv) / (1024 * 1024 * time_delta)
        
        # Cập nhật giá trị cuối
        self.last_io = (current_io, current_time)
        
        # Lưu vào lịch sử
        self.bandwidth_usage["sent"].append(sent)
        self.bandwidth_usage["received"].append(received)
        self.bandwidth_usage["timestamps"].append(datetime.now())
        
        return {"sent": sent, "received": received}
    
    def measure(self) -> float:
        """
        Đo độ trễ tổng hợp của mạng và lưu lại.
        Sử dụng Google DNS làm máy chủ tham chiếu.
        
        Returns:
            Độ trễ đo được (ms)
        """
        latency = self.measure_latency()
        self.add_value(latency)
        return latency
    
    def get_network_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê tổng hợp về hiệu suất mạng.
        
        Returns:
            Dictionary chứa thống kê mạng
        """
        # Đo băng thông hiện tại
        bandwidth = self.measure_bandwidth()
        
        # Lấy thông tin các kết nối mạng
        connections = len(psutil.net_connections())
        
        # Lấy thông tin IO counter
        io = psutil.net_io_counters()
        
        # Tổng hợp thông tin
        return {
            "latencies": self.server_latencies,
            "bandwidth": bandwidth,
            "connections": connections,
            "total_sent": io.bytes_sent / (1024 * 1024),  # MB
            "total_received": io.bytes_recv / (1024 * 1024),  # MB
            "packets_sent": io.packets_sent,
            "packets_recv": io.packets_recv,
            "err_in": io.errin,
            "err_out": io.errout,
            "drop_in": io.dropin,
            "drop_out": io.dropout
        }

class APIMetric(SystemMetric):
    """Chỉ số về hiệu suất API."""
    
    def __init__(self, api_name: str):
        """
        Khởi tạo chỉ số API.
        
        Args:
            api_name: Tên của API
        """
        super().__init__(
            name=f"{api_name}_api_latency",
            description=f"Độ trễ phản hồi của API {api_name}",
            unit="ms"
        )
        self.api_name = api_name
        self.success_count = 0
        self.error_count = 0
        self.status_codes = {}
        self.endpoints = {}
    
    def record_request(self, endpoint: str, latency: float, status_code: int, success: bool) -> None:
        """
        Ghi lại thông tin về một request API.
        
        Args:
            endpoint: Đường dẫn endpoint
            latency: Độ trễ phản hồi (ms)
            status_code: Mã trạng thái HTTP
            success: Yêu cầu thành công hay không
        """
        # Lưu độ trễ
        self.add_value(latency)
        
        # Cập nhật số lượng thành công/lỗi
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        # Cập nhật số lượng theo mã trạng thái
        if status_code in self.status_codes:
            self.status_codes[status_code] += 1
        else:
            self.status_codes[status_code] = 1
        
        # Cập nhật thông tin theo endpoint
        if endpoint not in self.endpoints:
            self.endpoints[endpoint] = {
                "count": 0,
                "latencies": [],
                "success": 0,
                "error": 0
            }
        
        # Cập nhật thông tin endpoint
        self.endpoints[endpoint]["count"] += 1
        self.endpoints[endpoint]["latencies"].append(latency)
        
        if success:
            self.endpoints[endpoint]["success"] += 1
        else:
            self.endpoints[endpoint]["error"] += 1
    
    def measure_request(self, endpoint: str, func: Callable, *args, **kwargs) -> Tuple[Any, float, int, bool]:
        """
        Đo độ trễ của một request API và ghi lại kết quả.
        
        Args:
            endpoint: Đường dẫn endpoint
            func: Hàm thực hiện request
            *args, **kwargs: Tham số cho hàm
            
        Returns:
            Tuple (kết quả, độ trễ, mã trạng thái, thành công)
        """
        start_time = time.time()
        result = None
        status_code = 0
        success = False
        
        try:
            # Thực hiện request
            result = func(*args, **kwargs)
            
            # Xác định mã trạng thái và thành công
            # Giả định result có thuộc tính status_code hoặc là tuple (data, status_code)
            if hasattr(result, 'status_code'):
                status_code = result.status_code
                success = 200 <= status_code < 300
            elif isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], int):
                status_code = result[1]
                success = 200 <= status_code < 300
            else:
                # Nếu không có mã trạng thái, giả định thành công
                status_code = 200
                success = True
                
        except Exception as e:
            # Xử lý lỗi
            if hasattr(e, 'status_code'):
                status_code = e.status_code
            elif hasattr(e, 'code'):
                status_code = e.code
            else:
                status_code = 500
            
            success = False
            
        finally:
            # Tính độ trễ
            latency = (time.time() - start_time) * 1000  # Chuyển sang ms
            
            # Ghi lại thông tin
            self.record_request(endpoint, latency, status_code, success)
            
            return result, latency, status_code, success
    
    def get_api_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê tổng hợp về hiệu suất API.
        
        Returns:
            Dictionary chứa thống kê API
        """
        total_requests = self.success_count + self.error_count
        
        # Tính toán tỷ lệ thành công
        success_rate = 0.0
        if total_requests > 0:
            success_rate = (self.success_count / total_requests) * 100
        
        # Tính toán độ trễ trung bình cho từng endpoint
        endpoint_stats = {}
        for endpoint, data in self.endpoints.items():
            if data["latencies"]:
                avg_latency = sum(data["latencies"]) / len(data["latencies"])
                success_rate_ep = 0.0
                if data["count"] > 0:
                    success_rate_ep = (data["success"] / data["count"]) * 100
                
                endpoint_stats[endpoint] = {
                    "count": data["count"],
                    "avg_latency": avg_latency,
                    "success_rate": success_rate_ep
                }
        
        return {
            "api_name": self.api_name,
            "total_requests": total_requests,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "status_codes": self.status_codes,
            "avg_latency": self.get_average(),
            "max_latency": self.get_max(),
            "min_latency": self.get_min(),
            "endpoints": endpoint_stats
        }
    
    def measure(self) -> float:
        """
        Hàm này chỉ trả về độ trễ trung bình hiện tại.
        Không thực sự đo gì mới, chỉ là phương thức giả để tuân thủ giao diện SystemMetric.
        
        Returns:
            Độ trễ trung bình
        """
        return self.get_average()

class ExchangeAPIMetric(APIMetric):
    """Chỉ số đặc biệt cho các API sàn giao dịch."""
    
    def __init__(self, exchange_name: str):
        """
        Khởi tạo chỉ số API sàn giao dịch.
        
        Args:
            exchange_name: Tên sàn giao dịch
        """
        super().__init__(api_name=exchange_name)
        self.rate_limit_hits = 0
        self.authentication_errors = 0
        self.timeout_errors = 0
        self.markets = {}
    
    def record_rate_limit_hit(self) -> None:
        """Ghi lại một lần vượt quá giới hạn tần suất."""
        self.rate_limit_hits += 1
    
    def record_authentication_error(self) -> None:
        """Ghi lại một lỗi xác thực."""
        self.authentication_errors += 1
    
    def record_timeout_error(self) -> None:
        """Ghi lại một lỗi timeout."""
        self.timeout_errors += 1
    
    def record_market_request(self, market: str, endpoint: str, latency: float, status_code: int, success: bool) -> None:
        """
        Ghi lại thông tin về một request API cho một thị trường cụ thể.
        
        Args:
            market: Tên thị trường (cặp tiền tệ)
            endpoint: Đường dẫn endpoint
            latency: Độ trễ phản hồi (ms)
            status_code: Mã trạng thái HTTP
            success: Yêu cầu thành công hay không
        """
        # Ghi lại thông tin cơ bản
        self.record_request(endpoint, latency, status_code, success)
        
        # Cập nhật thông tin theo market
        if market not in self.markets:
            self.markets[market] = {
                "count": 0,
                "latencies": [],
                "success": 0,
                "error": 0,
                "endpoints": {}
            }
        
        # Cập nhật thông tin market
        self.markets[market]["count"] += 1
        self.markets[market]["latencies"].append(latency)
        
        if success:
            self.markets[market]["success"] += 1
        else:
            self.markets[market]["error"] += 1
        
        # Cập nhật thông tin endpoint trong market
        if endpoint not in self.markets[market]["endpoints"]:
            self.markets[market]["endpoints"][endpoint] = {
                "count": 0,
                "latencies": [],
                "success": 0,
                "error": 0
            }
        
        # Cập nhật thông tin endpoint
        self.markets[market]["endpoints"][endpoint]["count"] += 1
        self.markets[market]["endpoints"][endpoint]["latencies"].append(latency)
        
        if success:
            self.markets[market]["endpoints"][endpoint]["success"] += 1
        else:
            self.markets[market]["endpoints"][endpoint]["error"] += 1
    
    def get_exchange_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê tổng hợp về hiệu suất API sàn giao dịch.
        
        Returns:
            Dictionary chứa thống kê API sàn giao dịch
        """
        # Lấy thống kê API cơ bản
        base_stats = self.get_api_stats()
        
        # Tính toán thống kê cho từng thị trường
        market_stats = {}
        for market, data in self.markets.items():
            if data["latencies"]:
                avg_latency = sum(data["latencies"]) / len(data["latencies"])
                success_rate_market = 0.0
                if data["count"] > 0:
                    success_rate_market = (data["success"] / data["count"]) * 100
                
                # Tính toán cho từng endpoint trong thị trường
                endpoint_stats = {}
                for endpoint, ep_data in data["endpoints"].items():
                    if ep_data["latencies"]:
                        ep_avg_latency = sum(ep_data["latencies"]) / len(ep_data["latencies"])
                        ep_success_rate = 0.0
                        if ep_data["count"] > 0:
                            ep_success_rate = (ep_data["success"] / ep_data["count"]) * 100
                        
                        endpoint_stats[endpoint] = {
                            "count": ep_data["count"],
                            "avg_latency": ep_avg_latency,
                            "success_rate": ep_success_rate
                        }
                
                market_stats[market] = {
                    "count": data["count"],
                    "avg_latency": avg_latency,
                    "success_rate": success_rate_market,
                    "endpoints": endpoint_stats
                }
        
        # Thêm thông tin về lỗi đặc biệt
        additional_stats = {
            "rate_limit_hits": self.rate_limit_hits,
            "authentication_errors": self.authentication_errors,
            "timeout_errors": self.timeout_errors,
            "markets": market_stats
        }
        
        # Kết hợp các thống kê
        return {**base_stats, **additional_stats}

class SystemMetricsCollector:
    """
    Lớp thu thập và quản lý các chỉ số hệ thống.
    Điều phối việc thu thập và báo cáo từ nhiều loại chỉ số khác nhau.
    """
    
    def __init__(self, name: str = "system_metrics"):
        """
        Khởi tạo bộ thu thập chỉ số.
        
        Args:
            name: Tên thành phần cho logger
        """
        self.name = name
        self.logger = get_system_logger(name)
        self.system_config = get_system_config()
        
        # Khởi tạo các chỉ số hệ thống cơ bản
        self.metrics = {
            "cpu": CPUMetric(),
            "memory": MemoryMetric(),
            "disk": DiskMetric(),
            "network": NetworkMetric()
        }
        
        # Dictionary để lưu trữ các chỉ số API
        self.api_metrics = {}
        
        # Dictionary để lưu trữ các chỉ số sàn giao dịch
        self.exchange_metrics = {}
        
        # Cờ để kiểm soát quá trình thu thập
        self.collecting = False
        self.collection_thread = None
        self.collection_interval = 60  # Giây
    
    def add_api_metric(self, api_name: str) -> APIMetric:
        """
        Thêm một chỉ số API mới.
        
        Args:
            api_name: Tên của API
            
        Returns:
            Đối tượng APIMetric đã tạo
        """
        if api_name not in self.api_metrics:
            self.api_metrics[api_name] = APIMetric(api_name)
        return self.api_metrics[api_name]
    
    def add_exchange_metric(self, exchange_name: str) -> ExchangeAPIMetric:
        """
        Thêm một chỉ số sàn giao dịch mới.
        
        Args:
            exchange_name: Tên sàn giao dịch
            
        Returns:
            Đối tượng ExchangeAPIMetric đã tạo
        """
        if exchange_name not in self.exchange_metrics:
            self.exchange_metrics[exchange_name] = ExchangeAPIMetric(exchange_name)
        return self.exchange_metrics[exchange_name]
    
    def collect_all_metrics(self) -> Dict[str, float]:
        """
        Thu thập tất cả các chỉ số hệ thống cơ bản.
        
        Returns:
            Dictionary chứa các chỉ số đã thu thập
        """
        results = {}
        
        # Thu thập CPU
        results["cpu"] = self.metrics["cpu"].measure()
        
        # Thu thập Memory
        results["memory"] = self.metrics["memory"].measure()
        
        # Thu thập Disk
        results["disk"] = self.metrics["disk"].measure()
        
        # Thu thập Network
        results["network"] = self.metrics["network"].measure()
        
        return results
    
    def _collection_loop(self) -> None:
        """
        Vòng lặp liên tục thu thập chỉ số hệ thống.
        Chạy trong một thread riêng.
        """
        self.logger.info("Bắt đầu thu thập chỉ số hệ thống")
        
        while self.collecting:
            try:
                # Thu thập chỉ số
                metrics = self.collect_all_metrics()
                
                # Ghi log
                self.logger.info(
                    f"Chỉ số hệ thống: CPU={metrics['cpu']:.1f}%, Mem={metrics['memory']:.1f}%, "
                    f"Disk={metrics['disk']:.1f}%, Network={metrics['network']:.1f}ms"
                )
                
                # Kiểm tra ngưỡng cảnh báo
                self._check_warning_thresholds(metrics)
                
            except Exception as e:
                self.logger.error(f"Lỗi khi thu thập chỉ số hệ thống: {str(e)}", exc_info=True)
            
            # Tạm dừng
            time.sleep(self.collection_interval)
        
        self.logger.info("Kết thúc thu thập chỉ số hệ thống")
    
    def _check_warning_thresholds(self, metrics: Dict[str, float]) -> None:
        """
        Kiểm tra các ngưỡng cảnh báo và ghi log nếu vượt quá.
        
        Args:
            metrics: Các chỉ số đã thu thập
        """
        # Kiểm tra CPU
        if metrics["cpu"] > 90:
            self.logger.warning(
                f"Cảnh báo: Sử dụng CPU cao ({metrics['cpu']:.1f}%)",
                extra={"component": self.name}
            )
        
        # Kiểm tra Memory
        if metrics["memory"] > 90:
            self.logger.warning(
                f"Cảnh báo: Sử dụng bộ nhớ cao ({metrics['memory']:.1f}%)",
                extra={"component": self.name}
            )
        
        # Kiểm tra Disk
        if metrics["disk"] > 90:
            self.logger.warning(
                f"Cảnh báo: Sử dụng đĩa cao ({metrics['disk']:.1f}%)",
                extra={"component": self.name}
            )
        
        # Kiểm tra Network
        if metrics["network"] > MAX_LATENCY_MS * 0.8:
            self.logger.warning(
                f"Cảnh báo: Độ trễ mạng cao ({metrics['network']:.1f}ms)",
                extra={"component": self.name}
            )
    
    def start_collection(self, interval: int = 60) -> None:
        """
        Bắt đầu thu thập chỉ số tự động trong một thread riêng.
        
        Args:
            interval: Khoảng thời gian giữa các lần thu thập (giây)
        """
        if self.collecting:
            self.logger.warning("Thu thập chỉ số đã đang chạy")
            return
        
        self.collection_interval = interval
        self.collecting = True
        
        # Tạo và khởi động thread
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            name=f"{self.name}_collector",
            daemon=True
        )
        self.collection_thread.start()
        
        self.logger.info(f"Đã bắt đầu thu thập chỉ số với chu kỳ {interval} giây")
    
    def stop_collection(self) -> None:
        """Dừng thu thập chỉ số tự động."""
        if not self.collecting:
            return
        
        self.collecting = False
        
        # Chờ thread kết thúc (tối đa 5 giây)
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        
        self.logger.info("Đã dừng thu thập chỉ số")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """
        Lấy bản tóm tắt về tình trạng hệ thống.
        
        Returns:
            Dictionary chứa thông tin tóm tắt hệ thống
        """
        # Thu thập chỉ số mới nhất
        self.collect_all_metrics()
        
        # CPU
        cpu = self.metrics["cpu"]
        cpu_info = {
            "usage": cpu.get_latest()[0],
            "average": cpu.get_average(10),
            "per_core": cpu.get_per_core_usage(),
            "load_avg": cpu.get_load_average()
        }
        
        # Memory
        memory = self.metrics["memory"]
        memory_info = {
            "usage": memory.get_latest()[0],
            "average": memory.get_average(10),
            "details": memory.get_memory_details(),
            "process": memory.measure_process_memory()
        }
        
        # Disk
        disk = self.metrics["disk"]
        disk_info = {
            "usage": disk.get_latest()[0],
            "average": disk.get_average(10),
            "details": disk.get_disk_details(),
            "io_stats": disk.get_io_stats()
        }
        
        # Network
        network = self.metrics["network"]
        network_info = {
            "latency": network.get_latest()[0],
            "average": network.get_average(10),
            "stats": network.get_network_stats()
        }
        
        # Thông tin về API (nếu có)
        api_info = {}
        for name, api_metric in self.api_metrics.items():
            api_info[name] = api_metric.get_api_stats()
        
        # Thông tin về sàn giao dịch (nếu có)
        exchange_info = {}
        for name, exchange_metric in self.exchange_metrics.items():
            exchange_info[name] = exchange_metric.get_exchange_stats()
        
        # Thông tin hệ thống
        system_info = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "uptime": self._get_uptime(),
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "system": system_info,
            "cpu": cpu_info,
            "memory": memory_info,
            "disk": disk_info,
            "network": network_info,
            "api": api_info,
            "exchange": exchange_info
        }
    
    def _get_uptime(self) -> float:
        """
        Lấy thời gian hoạt động của hệ thống.
        
        Returns:
            Số giây hệ thống đã hoạt động
        """
        try:
            return time.time() - psutil.boot_time()
        except (AttributeError, OSError):
            return 0.0
    
    def log_system_summary(self) -> None:
        """Ghi log tóm tắt về tình trạng hệ thống."""
        try:
            summary = self.get_system_summary()
            
            # Ghi log với định dạng gọn
            self.logger.info(
                f"Tóm tắt hệ thống: CPU={summary['cpu']['usage']:.1f}%, "
                f"Mem={summary['memory']['usage']:.1f}%, "
                f"Disk={summary['disk']['usage']:.1f}%, "
                f"Net={summary['network']['latency']:.1f}ms, "
                f"Uptime={summary['system']['uptime']/3600:.1f}h"
            )
            
            # Ghi log chi tiết dưới dạng JSON
            self.logger.debug(f"Chi tiết hệ thống: {json.dumps(summary, ensure_ascii=False)}")
            
            # Nếu có bất kỳ API nào có tỷ lệ lỗi cao, ghi log cảnh báo
            for api_name, api_data in summary["api"].items():
                if api_data["success_rate"] < 90 and api_data["total_requests"] > 10:
                    self.logger.warning(
                        f"API {api_name} có tỷ lệ thành công thấp: {api_data['success_rate']:.1f}%",
                        extra={"component": self.name}
                    )
            
            # Tương tự cho các sàn giao dịch
            for ex_name, ex_data in summary["exchange"].items():
                if ex_data["success_rate"] < 90 and ex_data["total_requests"] > 10:
                    self.logger.warning(
                        f"Sàn {ex_name} có tỷ lệ thành công thấp: {ex_data['success_rate']:.1f}%",
                        extra={"component": self.name}
                    )
                
                # Kiểm tra rate limit
                if ex_data["rate_limit_hits"] > 0:
                    self.logger.warning(
                        f"Sàn {ex_name} đã vượt quá giới hạn tần suất {ex_data['rate_limit_hits']} lần",
                        extra={"component": self.name}
                    )
            
        except Exception as e:
            self.logger.error(f"Lỗi khi ghi log tóm tắt hệ thống: {str(e)}", exc_info=True)
    
    def export_metrics_to_csv(self, filepath: str) -> bool:
        """
        Xuất dữ liệu chỉ số ra file CSV.
        
        Args:
            filepath: Đường dẫn file CSV
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        try:
            # Chuẩn bị dữ liệu cho từng chỉ số
            dataframes = []
            
            # Chỉ số hệ thống
            for name, metric in self.metrics.items():
                df = metric.to_dataframe()
                df["metric_name"] = name
                dataframes.append(df)
            
            # Chỉ số API
            for name, metric in self.api_metrics.items():
                df = metric.to_dataframe()
                df["metric_name"] = f"api_{name}"
                dataframes.append(df)
            
            # Chỉ số sàn giao dịch
            for name, metric in self.exchange_metrics.items():
                df = metric.to_dataframe()
                df["metric_name"] = f"exchange_{name}"
                dataframes.append(df)
            
            # Kết hợp tất cả và lưu
            if dataframes:
                combined_df = pd.concat(dataframes)
                combined_df.to_csv(filepath, index=False)
                self.logger.info(f"Đã xuất chỉ số ra file {filepath}")
                return True
            else:
                self.logger.warning("Không có dữ liệu chỉ số để xuất")
                return False
                
        except Exception as e:
            self.logger.error(f"Lỗi khi xuất chỉ số ra file CSV: {str(e)}", exc_info=True)
            return False
    
    def reset_all_metrics(self) -> None:
        """Đặt lại tất cả các chỉ số."""
        # Đặt lại chỉ số hệ thống
        for metric in self.metrics.values():
            metric.reset()
        
        # Đặt lại chỉ số API
        for metric in self.api_metrics.values():
            metric.reset()
        
        # Đặt lại chỉ số sàn giao dịch
        for metric in self.exchange_metrics.values():
            metric.reset()
        
        self.logger.info("Đã đặt lại tất cả các chỉ số hệ thống")

# Tạo instance mặc định để sử dụng trong ứng dụng
system_metrics = SystemMetricsCollector()

def get_system_metrics() -> SystemMetricsCollector:
    """
    Hàm helper để lấy instance SystemMetricsCollector.
    
    Returns:
        Instance SystemMetricsCollector mặc định
    """
    return system_metrics

# API Decorator để đo hiệu suất API
def measure_api_performance(api_name: str, endpoint: str):
    """
    Decorator để đo hiệu suất API.
    
    Args:
        api_name: Tên của API
        endpoint: Tên endpoint
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Lấy metrics collector
            metrics_collector = get_system_metrics()
            
            # Lấy hoặc tạo API metric
            if api_name.lower() in ["binance", "bybit", "okex", "coinbase", "kraken"]:
                # Đây là sàn giao dịch
                api_metric = metrics_collector.add_exchange_metric(api_name)
            else:
                # API thông thường
                api_metric = metrics_collector.add_api_metric(api_name)
            
            # Đo hiệu suất
            start_time = time.time()
            result = None
            status_code = 0
            success = False
            
            try:
                # Thực hiện hàm
                result = func(*args, **kwargs)
                
                # Xác định thành công và mã trạng thái
                if hasattr(result, 'status_code'):
                    status_code = result.status_code
                    success = 200 <= status_code < 300
                elif isinstance(result, tuple) and len(result) >= 2:
                    # Giả định kết quả là (data, status_code)
                    result, status_code = result[0], result[1]
                    success = 200 <= status_code < 300
                else:
                    # Giả định thành công nếu không có mã trạng thái
                    status_code = 200
                    success = True
                
            except Exception as e:
                # Xử lý lỗi
                if hasattr(e, 'status_code'):
                    status_code = e.status_code
                elif hasattr(e, 'code'):
                    status_code = e.code
                else:
                    status_code = 500
                
                success = False
                
                # Xử lý lỗi đặc biệt cho sàn giao dịch
                if isinstance(api_metric, ExchangeAPIMetric):
                    if "rate limit" in str(e).lower():
                        api_metric.record_rate_limit_hit()
                    elif "auth" in str(e).lower() or "key" in str(e).lower():
                        api_metric.record_authentication_error()
                    elif "timeout" in str(e).lower() or "time out" in str(e).lower():
                        api_metric.record_timeout_error()
                
                # Ném lại ngoại lệ
                raise
                
            finally:
                # Tính độ trễ
                latency = (time.time() - start_time) * 1000  # Chuyển sang ms
                
                # Ghi lại chỉ số
                if isinstance(api_metric, ExchangeAPIMetric) and len(args) > 0 and isinstance(args[0], str):
                    # Nếu tham số đầu tiên là chuỗi, giả định đó là symbol/market
                    api_metric.record_market_request(args[0], endpoint, latency, status_code, success)
                else:
                    api_metric.record_request(endpoint, latency, status_code, success)
            
            return result
        return wrapper
    return decorator

import functools

# Hàm decorator đo thời gian thực thi
def measure_execution_time(logger: logging.Logger = None, level: int = logging.DEBUG):
    """
    Decorator để đo thời gian thực thi của một hàm.
    
    Args:
        logger: Logger để ghi log kết quả (mặc định là None)
        level: Mức độ log (mặc định là DEBUG)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # ms
            
            # Ghi log nếu cần
            if logger:
                logger.log(
                    level,
                    f"Thời gian thực thi {func.__name__}: {execution_time:.2f}ms"
                )
            
            return result
        return wrapper
    return decorator