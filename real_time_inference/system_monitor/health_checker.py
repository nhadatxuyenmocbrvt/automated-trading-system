"""
Kiểm tra sức khỏe hệ thống.
File này cung cấp các lớp và hàm để theo dõi trạng thái hoạt động của hệ thống giao dịch,
bao gồm việc giám sát tài nguyên hệ thống, kết nối mạng, và trạng thái các thành phần quan trọng.
"""

import os
import psutil
import socket
import logging
import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from config.constants import SystemStatus
from logs.logger import SystemLogger

class SystemHealthChecker:
    """
    Lớp kiểm tra sức khỏe hệ thống.
    Giám sát tài nguyên hệ thống như CPU, RAM, disk và mạng. Đồng thời
    kiểm tra trạng thái của các thành phần chính trong hệ thống giao dịch.
    """
    
    def __init__(
        self, 
        check_interval: float = 60.0,
        cpu_threshold: float = 80.0, 
        memory_threshold: float = 80.0,
        disk_threshold: float = 90.0,
        max_response_time: float = 2.0,
        external_endpoints: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo kiểm tra sức khỏe hệ thống.
        
        Args:
            check_interval: Khoảng thời gian giữa các lần kiểm tra (giây)
            cpu_threshold: Ngưỡng % sử dụng CPU để cảnh báo
            memory_threshold: Ngưỡng % sử dụng RAM để cảnh báo
            disk_threshold: Ngưỡng % sử dụng ổ đĩa để cảnh báo
            max_response_time: Thời gian phản hồi tối đa chấp nhận được (giây)
            external_endpoints: Danh sách các endpoint bên ngoài để kiểm tra
            logger: Logger tùy chỉnh
        """
        # Logger
        self.logger = logger or SystemLogger("health_checker")
        
        # Cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Ngưỡng cảnh báo
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
        self.max_response_time = max_response_time
        self.check_interval = check_interval
        
        # Danh sách endpoint kiểm tra
        self.external_endpoints = external_endpoints or [
            "api.binance.com",
            "api.bybit.com",
            "api.coinmarketcap.com",
            "api.coingecko.com"
        ]
        
        # Callbacks khi phát hiện vấn đề
        self.alert_callbacks = []
        
        # Trạng thái 
        self.system_health = {
            "status": SystemStatus.NORMAL.value,
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "network_status": {},
            "components_status": {},
            "last_check_time": None,
            "issues": []
        }
        
        # Thread giám sát
        self.monitor_thread = None
        self.stop_flag = threading.Event()
        
        self.logger.info(f"Khởi tạo SystemHealthChecker với chu kỳ kiểm tra {check_interval} giây")
    
    def start_monitoring(self) -> bool:
        """
        Bắt đầu giám sát sức khỏe hệ thống trong thread riêng.
        
        Returns:
            True nếu bắt đầu thành công, False nếu không
        """
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            self.logger.warning("Giám sát sức khỏe hệ thống đã đang chạy")
            return False
        
        # Đặt lại cờ dừng
        self.stop_flag.clear()
        
        # Tạo và khởi động thread mới
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="HealthMonitorThread",
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("Đã bắt đầu giám sát sức khỏe hệ thống")
        return True
    
    def stop_monitoring(self) -> bool:
        """
        Dừng giám sát sức khỏe hệ thống.
        
        Returns:
            True nếu dừng thành công, False nếu không
        """
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.logger.warning("Giám sát sức khỏe hệ thống không đang chạy")
            return False
        
        # Đặt cờ dừng
        self.stop_flag.set()
        
        # Chờ thread kết thúc
        self.monitor_thread.join(timeout=5.0)
        
        if self.monitor_thread.is_alive():
            self.logger.warning("Không thể dừng thread giám sát sức khỏe hệ thống")
            return False
        
        self.monitor_thread = None
        self.logger.info("Đã dừng giám sát sức khỏe hệ thống")
        return True
    
    def _monitoring_loop(self) -> None:
        """
        Vòng lặp giám sát chính.
        """
        self.logger.info("Vòng lặp giám sát sức khỏe hệ thống bắt đầu")
        
        while not self.stop_flag.is_set():
            try:
                # Thực hiện kiểm tra sức khỏe
                self.check_system_health()
                
                # Ghi log trạng thái
                self._log_health_status()
                
                # Đợi đến lần kiểm tra tiếp theo
                self.stop_flag.wait(self.check_interval)
            except Exception as e:
                self.logger.error(f"Lỗi trong vòng lặp giám sát: {str(e)}", exc_info=True)
                # Đợi một chút trước khi thử lại
                time.sleep(5)
        
        self.logger.info("Vòng lặp giám sát sức khỏe hệ thống đã kết thúc")
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Thực hiện kiểm tra sức khỏe toàn diện của hệ thống.
        
        Returns:
            Dict chứa kết quả kiểm tra
        """
        # Đặt lại danh sách vấn đề
        self.system_health["issues"] = []
        
        # Kiểm tra tài nguyên hệ thống
        self._check_cpu_usage()
        self._check_memory_usage()
        self._check_disk_usage()
        
        # Kiểm tra kết nối mạng
        self._check_network_connectivity()
        
        # Kiểm tra các thành phần hệ thống
        self._check_components_status()
        
        # Cập nhật thời gian kiểm tra cuối
        self.system_health["last_check_time"] = datetime.now()
        
        # Xác định trạng thái tổng thể dựa trên số lượng vấn đề
        if len(self.system_health["issues"]) == 0:
            self.system_health["status"] = SystemStatus.NORMAL.value
        elif any(issue["severity"] == "critical" for issue in self.system_health["issues"]):
            self.system_health["status"] = SystemStatus.CRITICAL.value
        elif len(self.system_health["issues"]) > 2:
            self.system_health["status"] = SystemStatus.WARNING.value
        else:
            self.system_health["status"] = SystemStatus.CAUTION.value
        
        # Gọi các callback cảnh báo nếu có vấn đề
        if self.system_health["issues"] and self.alert_callbacks:
            for callback in self.alert_callbacks:
                try:
                    callback(self.system_health)
                except Exception as e:
                    self.logger.error(f"Lỗi khi gọi callback cảnh báo: {str(e)}", exc_info=True)
        
        return self.system_health
    
    def _check_cpu_usage(self) -> None:
        """
        Kiểm tra mức sử dụng CPU.
        """
        cpu_usage = psutil.cpu_percent(interval=1)
        self.system_health["cpu_usage"] = cpu_usage
        
        if cpu_usage > self.cpu_threshold:
            self.system_health["issues"].append({
                "type": "cpu",
                "message": f"Sử dụng CPU cao: {cpu_usage:.1f}% (ngưỡng: {self.cpu_threshold:.1f}%)",
                "value": cpu_usage,
                "threshold": self.cpu_threshold,
                "timestamp": datetime.now().isoformat(),
                "severity": "warning" if cpu_usage < 95 else "critical"
            })
    
    def _check_memory_usage(self) -> None:
        """
        Kiểm tra mức sử dụng bộ nhớ RAM.
        """
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        self.system_health["memory_usage"] = memory_usage
        
        if memory_usage > self.memory_threshold:
            self.system_health["issues"].append({
                "type": "memory",
                "message": f"Sử dụng RAM cao: {memory_usage:.1f}% (ngưỡng: {self.memory_threshold:.1f}%)",
                "value": memory_usage,
                "threshold": self.memory_threshold,
                "timestamp": datetime.now().isoformat(),
                "severity": "warning" if memory_usage < 95 else "critical"
            })
    
    def _check_disk_usage(self) -> None:
        """
        Kiểm tra mức sử dụng ổ đĩa.
        """
        # Lấy thư mục hiện tại
        current_dir = Path.cwd()
        
        # Kiểm tra ổ đĩa chứa thư mục hiện tại
        disk_usage = psutil.disk_usage(current_dir).percent
        self.system_health["disk_usage"] = disk_usage
        
        if disk_usage > self.disk_threshold:
            self.system_health["issues"].append({
                "type": "disk",
                "message": f"Sử dụng ổ đĩa cao: {disk_usage:.1f}% (ngưỡng: {self.disk_threshold:.1f}%)",
                "value": disk_usage,
                "threshold": self.disk_threshold,
                "timestamp": datetime.now().isoformat(),
                "severity": "warning" if disk_usage < 98 else "critical"
            })
    
    def _check_network_connectivity(self) -> None:
        """
        Kiểm tra kết nối mạng đến các endpoint bên ngoài.
        """
        network_status = {}
        
        for endpoint in self.external_endpoints:
            try:
                # Bỏ protocol nếu có
                host = endpoint.split("://")[-1].split("/")[0]
                
                # Đo thời gian phản hồi
                start_time = time.time()
                socket.create_connection((host, 80), timeout=self.max_response_time)
                response_time = time.time() - start_time
                
                status = {
                    "status": "connected",
                    "response_time": response_time
                }
                
                # Nếu thời gian phản hồi quá cao
                if response_time > self.max_response_time:
                    self.system_health["issues"].append({
                        "type": "network",
                        "message": f"Kết nối chậm đến {endpoint}: {response_time:.2f}s",
                        "endpoint": endpoint,
                        "response_time": response_time,
                        "threshold": self.max_response_time,
                        "timestamp": datetime.now().isoformat(),
                        "severity": "warning"
                    })
            except (socket.timeout, socket.error, OSError) as e:
                status = {
                    "status": "disconnected",
                    "error": str(e)
                }
                
                self.system_health["issues"].append({
                    "type": "network",
                    "message": f"Không thể kết nối đến {endpoint}: {str(e)}",
                    "endpoint": endpoint,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "severity": "critical" if "api.binance.com" in endpoint or "api.bybit.com" in endpoint else "warning"
                })
            
            network_status[endpoint] = status
        
        self.system_health["network_status"] = network_status
    
    def _check_components_status(self) -> None:
        """
        Kiểm tra trạng thái của các thành phần chính trong hệ thống.
        """
        components_status = {}
        
        # Kiểm tra xem các thành phần cần thiết có đang chạy không
        # (Ví dụ: các process, service, thread, etc.)
        
        # Kiểm tra các tiến trình Python đang chạy
        python_processes = [p for p in psutil.process_iter() if 'python' in p.name().lower()]
        
        # Kiểm tra tiến trình chính (giả định rằng nó có tên chứa 'trading')
        main_process_running = any('trading' in p.cmdline()[1] if len(p.cmdline()) > 1 else False 
                                   for p in python_processes)
        
        components_status["main_process"] = {
            "status": "running" if main_process_running else "stopped",
            "last_checked": datetime.now().isoformat()
        }
        
        if not main_process_running:
            self.system_health["issues"].append({
                "type": "component",
                "message": "Tiến trình giao dịch chính không hoạt động",
                "component": "main_process",
                "timestamp": datetime.now().isoformat(),
                "severity": "critical"
            })
        
        # Kiểm tra cơ sở dữ liệu hoặc các dịch vụ khác nếu cần...
        
        self.system_health["components_status"] = components_status
    
    def _log_health_status(self) -> None:
        """
        Ghi log trạng thái sức khỏe hệ thống.
        """
        status = self.system_health["status"]
        
        # Ghi log với mức độ phù hợp với trạng thái
        if status == SystemStatus.NORMAL.value:
            self.logger.info(f"Trạng thái hệ thống: {status}, CPU: {self.system_health['cpu_usage']:.1f}%, "
                           f"RAM: {self.system_health['memory_usage']:.1f}%, "
                           f"Ổ đĩa: {self.system_health['disk_usage']:.1f}%")
        elif status == SystemStatus.CAUTION.value:
            self.logger.warning(f"Trạng thái hệ thống: {status}, CPU: {self.system_health['cpu_usage']:.1f}%, "
                              f"RAM: {self.system_health['memory_usage']:.1f}%, "
                              f"Ổ đĩa: {self.system_health['disk_usage']:.1f}%, "
                              f"Vấn đề: {len(self.system_health['issues'])}")
        else:
            self.logger.error(f"Trạng thái hệ thống: {status}, CPU: {self.system_health['cpu_usage']:.1f}%, "
                            f"RAM: {self.system_health['memory_usage']:.1f}%, "
                            f"Ổ đĩa: {self.system_health['disk_usage']:.1f}%, "
                            f"Vấn đề: {len(self.system_health['issues'])}")
            
            # Log chi tiết các vấn đề
            for issue in self.system_health["issues"]:
                if issue["severity"] == "critical":
                    self.logger.critical(f"Vấn đề nghiêm trọng: {issue['message']}")
                else:
                    self.logger.warning(f"Cảnh báo: {issue['message']}")
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Đăng ký callback để nhận thông báo khi có vấn đề.
        
        Args:
            callback: Hàm callback nhận dict trạng thái sức khỏe
        """
        if callback not in self.alert_callbacks:
            self.alert_callbacks.append(callback)
            self.logger.debug(f"Đã đăng ký callback cảnh báo: {callback.__name__}")
    
    def unregister_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Hủy đăng ký callback cảnh báo.
        
        Args:
            callback: Hàm callback cần hủy
            
        Returns:
            True nếu hủy thành công, False nếu không tìm thấy
        """
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            self.logger.debug(f"Đã hủy đăng ký callback cảnh báo: {callback.__name__}")
            return True
        return False
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Lấy bản tóm tắt sức khỏe hệ thống hiện tại.
        
        Returns:
            Dict chứa thông tin tóm tắt sức khỏe
        """
        return {
            "status": self.system_health["status"],
            "cpu_usage": self.system_health["cpu_usage"],
            "memory_usage": self.system_health["memory_usage"],
            "disk_usage": self.system_health["disk_usage"],
            "issue_count": len(self.system_health["issues"]),
            "last_check_time": self.system_health["last_check_time"],
            "network_ok": all(s["status"] == "connected" for s in self.system_health["network_status"].values()) 
                         if self.system_health["network_status"] else False
        }
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """
        Chạy chẩn đoán hệ thống chi tiết khi có vấn đề.
        
        Returns:
            Dict kết quả chẩn đoán chi tiết
        """
        # Chạy kiểm tra ngay lập tức
        self.check_system_health()
        
        # Thu thập thông tin hệ thống chi tiết
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "platform": os.name,
                "cpu_count": psutil.cpu_count(logical=True),
                "physical_cpu_count": psutil.cpu_count(logical=False),
                "total_memory": psutil.virtual_memory().total,
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            },
            "processes": [],
            "network_details": {},
            "health_status": self.system_health
        }
        
        # Thu thập thông tin về các tiến trình
        for proc in psutil.process_iter(['pid', 'name', 'username', 'memory_percent', 'cpu_percent']):
            try:
                pinfo = proc.as_dict()
                # Chỉ lấy các tiến trình Python hoặc tiêu thụ nhiều tài nguyên
                if ('python' in pinfo['name'].lower() or 
                    pinfo['memory_percent'] > 1.0 or 
                    pinfo['cpu_percent'] > 5.0):
                    # Bổ sung thông tin lệnh
                    try:
                        pinfo['cmdline'] = proc.cmdline()
                    except:
                        pinfo['cmdline'] = []
                    
                    # Bổ sung thông tin thời gian chạy
                    try:
                        pinfo['running_time'] = str(datetime.now() - 
                                                   datetime.fromtimestamp(proc.create_time()))
                    except:
                        pinfo['running_time'] = "unknown"
                    
                    diagnostics["processes"].append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # Chi tiết về kết nối mạng
        try:
            # Lấy thông tin về các kết nối mạng hiện tại
            connections = psutil.net_connections()
            established_conns = [c for c in connections if c.status == 'ESTABLISHED']
            
            # Thống kê theo port
            port_stats = {}
            for conn in established_conns:
                if conn.laddr.port not in port_stats:
                    port_stats[conn.laddr.port] = 0
                port_stats[conn.laddr.port] += 1
            
            diagnostics["network_details"] = {
                "total_connections": len(connections),
                "established_connections": len(established_conns),
                "connections_by_port": port_stats,
                "net_io_counters": psutil.net_io_counters()._asdict()
            }
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập thông tin mạng chi tiết: {str(e)}", exc_info=True)
        
        # Ghi log kết quả chẩn đoán
        self.logger.info(f"Đã chạy chẩn đoán hệ thống: {len(diagnostics['processes'])} tiến trình, "
                        f"{diagnostics['network_details'].get('total_connections', 0)} kết nối mạng")
        
        return diagnostics
    
    def get_system_status(self) -> str:
        """
        Lấy trạng thái hiện tại của hệ thống.
        
        Returns:
            Trạng thái hệ thống (NORMAL, CAUTION, WARNING, CRITICAL)
        """
        return self.system_health["status"]
    
    def save_health_data(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Lưu dữ liệu sức khỏe hệ thống hiện tại vào file.
        
        Args:
            file_path: Đường dẫn file (None để sử dụng đường dẫn mặc định)
            
        Returns:
            True nếu lưu thành công, False nếu không
        """
        if file_path is None:
            # Tạo tên file dựa trên thời gian
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = Path(f"logs/system_health_{timestamp}.json")
        
        try:
            # Đảm bảo thư mục tồn tại
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Chuyển đổi datetime thành chuỗi
            health_data = self.system_health.copy()
            if health_data["last_check_time"] is not None:
                health_data["last_check_time"] = health_data["last_check_time"].isoformat()
            
            # Lưu vào file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(health_data, f, ensure_ascii=False, indent=4)
            
            self.logger.info(f"Đã lưu dữ liệu sức khỏe hệ thống vào {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu dữ liệu sức khỏe hệ thống: {str(e)}", exc_info=True)
            return False
    
    def get_available_resources(self) -> Dict[str, Any]:
        """
        Lấy thông tin về tài nguyên khả dụng trên hệ thống.
        
        Returns:
            Dict chứa thông tin tài nguyên khả dụng
        """
        resources = {
            "cpu": {
                "available_percent": 100.0 - psutil.cpu_percent(interval=0.1),
                "cores": psutil.cpu_count(logical=True)
            },
            "memory": {
                "available_bytes": psutil.virtual_memory().available,
                "available_percent": 100.0 - psutil.virtual_memory().percent
            },
            "disk": {
                "available_bytes": psutil.disk_usage(Path.cwd()).free,
                "available_percent": 100.0 - psutil.disk_usage(Path.cwd()).percent
            }
        }
        
        return resources

# Singleton instance
_health_checker_instance = None

def get_health_checker() -> SystemHealthChecker:
    """
    Lấy instance singleton của SystemHealthChecker.
    
    Returns:
        Instance SystemHealthChecker
    """
    global _health_checker_instance
    if _health_checker_instance is None:
        _health_checker_instance = SystemHealthChecker()
    return _health_checker_instance