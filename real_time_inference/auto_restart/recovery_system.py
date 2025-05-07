"""
Hệ thống phục hồi tự động.
File này cung cấp lớp RecoverySystem để khôi phục hệ thống sau các lỗi,
bao gồm việc lưu/khôi phục trạng thái và khởi động lại các thành phần.
"""

import os
import json
import time
import signal
import logging
import threading
import subprocess
import traceback
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime
from pathlib import Path

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from real_time_inference.auto_restart.error_handler import ErrorSeverity, ErrorCategory, RecoveryAction
from real_time_inference.system_monitor.notification_manager import NotificationManager, NotificationPriority, NotificationType

class ComponentStatus(Enum):
    """Trạng thái thành phần."""
    RUNNING = "running"       # Đang chạy
    STOPPED = "stopped"       # Đã dừng
    RESTARTING = "restarting" # Đang khởi động lại
    ERROR = "error"           # Lỗi
    UNKNOWN = "unknown"       # Không xác định

class RecoveryStrategy(Enum):
    """Chiến lược khôi phục."""
    IMMEDIATE = "immediate"   # Khôi phục ngay lập tức
    GRADUAL = "gradual"       # Khôi phục dần dần
    PRIORITIZED = "prioritized" # Khôi phục theo ưu tiên
    DEPENDENT = "dependent"   # Khôi phục theo phụ thuộc

class RecoverySystem:
    """
    Hệ thống phục hồi tự động.
    Cung cấp cơ chế để khôi phục hệ thống sau lỗi.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        state_dir: Optional[str] = None,
        notification_manager: Optional[NotificationManager] = None,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.GRADUAL,
        max_restart_attempts: int = 3,
        escalation_threshold: int = 3,
        cooldown_period: int = 300,  # 5 phút
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo RecoverySystem.
        
        Args:
            config_path: Đường dẫn file cấu hình
            state_dir: Thư mục lưu trạng thái
            notification_manager: Quản lý thông báo
            recovery_strategy: Chiến lược khôi phục
            max_restart_attempts: Số lần khởi động lại tối đa
            escalation_threshold: Ngưỡng leo thang (số lần lỗi liên tiếp)
            cooldown_period: Thời gian nghỉ giữa các lần khởi động lại (giây)
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("recovery_system")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config().get_all()
        
        # Lưu trữ thông tin cấu hình
        self.config_path = config_path
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()
        
        # Quản lý thông báo
        self.notification_manager = notification_manager
        
        # Thiết lập thư mục lưu trạng thái
        if state_dir is None:
            base_dir = os.path.join(self.system_config.get("data_dir", "data"), "system_state")
            self.state_dir = base_dir
        else:
            self.state_dir = state_dir
            
        # Đảm bảo thư mục tồn tại
        os.makedirs(self.state_dir, exist_ok=True)
        
        # Thiết lập các tham số
        self.recovery_strategy = recovery_strategy
        self.max_restart_attempts = max_restart_attempts
        self.escalation_threshold = escalation_threshold
        self.cooldown_period = cooldown_period
        
        # Trạng thái các thành phần
        self.component_status = {}
        self.restart_counts = {}
        self.last_restart_times = {}
        self.components_hierarchy = self._load_component_hierarchy()
        
        # Lịch sử khôi phục
        self.recovery_history = []
        
        # Theo dõi trạng thái hệ thống
        self.system_status = "normal"  # "normal", "degraded", "critical", "recovering"
        self.recovery_in_progress = False
        
        # Thiết lập lock thread
        self.lock = threading.Lock()
        
        # Khởi tạo danh sách thành phần từ cấu hình
        self._initialize_components()
        
        self.logger.info(f"Đã khởi tạo RecoverySystem với chiến lược {recovery_strategy.value}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Lấy cấu hình mặc định.
        
        Returns:
            Dict chứa cấu hình
        """
        return {
            "components": {
                "data_collector": {
                    "script_path": "data_collectors/market_data/realtime_data_stream.py",
                    "priority": 1,
                    "dependencies": [],
                    "startup_timeout": 30,
                    "restart_command": "python {script_path}",
                    "health_check": {
                        "type": "http",
                        "endpoint": "http://localhost:8001/health",
                        "timeout": 5
                    }
                },
                "inference_engine": {
                    "script_path": "real_time_inference/inference_engine.py",
                    "priority": 2,
                    "dependencies": ["data_collector"],
                    "startup_timeout": 60,
                    "restart_command": "python {script_path}",
                    "health_check": {
                        "type": "http",
                        "endpoint": "http://localhost:8002/health",
                        "timeout": 5
                    }
                },
                "trade_executor": {
                    "script_path": "deployment/trade_executor.py",
                    "priority": 3,
                    "dependencies": ["inference_engine"],
                    "startup_timeout": 45,
                    "restart_command": "python {script_path}",
                    "health_check": {
                        "type": "http",
                        "endpoint": "http://localhost:8003/health",
                        "timeout": 5
                    }
                },
                "system_monitor": {
                    "script_path": "real_time_inference/system_monitor/health_checker.py",
                    "priority": 0,
                    "dependencies": [],
                    "startup_timeout": 15,
                    "restart_command": "python {script_path}",
                    "health_check": {
                        "type": "process",
                        "process_name": "health_checker.py",
                        "timeout": 5
                    }
                }
            },
            "recovery": {
                "state_save_interval": 60,  # giây
                "default_strategy": "gradual",
                "emergency_contacts": ["admin@example.com"],
                "max_restart_attempts": 3,
                "escalation_threshold": 3,
                "cooldown_period": 300  # 5 phút
            }
        }
    
    def _initialize_components(self) -> None:
        """
        Khởi tạo trạng thái các thành phần.
        """
        components = self.config.get("components", {})
        
        for component_id, component_config in components.items():
            self.component_status[component_id] = ComponentStatus.UNKNOWN.value
            self.restart_counts[component_id] = 0
            self.last_restart_times[component_id] = 0
            
            self.logger.debug(f"Đã khởi tạo thành phần {component_id}")
    
    def _load_component_hierarchy(self) -> Dict[str, List[str]]:
        """
        Tải thông tin phân cấp các thành phần.
        
        Returns:
            Dict với key là component_id và value là danh sách các thành phần phụ thuộc
        """
        hierarchy = {}
        components = self.config.get("components", {})
        
        # Tạo danh sách các thành phần phụ thuộc
        for component_id, component_config in components.items():
            hierarchy[component_id] = []
            
        # Xác định các thành phần phụ thuộc
        for component_id, component_config in components.items():
            dependencies = component_config.get("dependencies", [])
            
            for dep in dependencies:
                if dep in hierarchy:
                    hierarchy[dep].append(component_id)
        
        return hierarchy
    
    def perform_recovery(
        self,
        error_id: str,
        component_id: str,
        recovery_action: Union[str, RecoveryAction],
        error_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Thực hiện khôi phục sau lỗi.
        
        Args:
            error_id: ID lỗi
            component_id: ID thành phần
            recovery_action: Hành động khôi phục
            error_details: Chi tiết lỗi
            
        Returns:
            Dict chứa kết quả khôi phục
        """
        # Chuyển đổi enum thành giá trị nếu cần
        if isinstance(recovery_action, RecoveryAction):
            recovery_action = recovery_action.value
            
        # Tạo bản ghi khôi phục
        recovery_record = {
            'id': f"REC{int(time.time())}",
            'error_id': error_id,
            'component_id': component_id,
            'action': recovery_action,
            'timestamp': datetime.now().isoformat(),
            'status': 'in_progress',
            'success': False,
            'details': {},
            'error_details': error_details
        }
        
        # Ghi log bắt đầu khôi phục
        self.logger.info(f"Bắt đầu khôi phục thành phần {component_id}, hành động: {recovery_action}")
        
        # Kiểm tra xem thành phần có tồn tại không
        if component_id not in self.component_status:
            recovery_record['status'] = 'failed'
            recovery_record['details']['reason'] = f"Thành phần {component_id} không tồn tại"
            self.logger.error(f"Không thể khôi phục: Thành phần {component_id} không tồn tại")
            
            with self.lock:
                self.recovery_history.append(recovery_record)
                
            return recovery_record
        
        # Kiểm tra xem có đang trong quá trình khôi phục không
        with self.lock:
            if self.recovery_in_progress:
                # Thêm vào thông tin chi tiết
                recovery_record['status'] = 'queued'
                recovery_record['details']['reason'] = "Đã có quá trình khôi phục khác đang diễn ra"
                self.logger.warning(f"Đã có quá trình khôi phục khác đang diễn ra, thêm vào hàng đợi")
                
                self.recovery_history.append(recovery_record)
                
                return recovery_record
                
            # Đánh dấu đang trong quá trình khôi phục
            self.recovery_in_progress = True
        
        try:
            # Thực hiện hành động khôi phục tương ứng
            if recovery_action == RecoveryAction.RETRY.value:
                result = self._retry_component(component_id, recovery_record)
                
            elif recovery_action == RecoveryAction.RESTART_COMPONENT.value:
                result = self._restart_component(component_id, recovery_record)
                
            elif recovery_action == RecoveryAction.RESTART_SYSTEM.value:
                result = self._restart_system(recovery_record)
                
            elif recovery_action == RecoveryAction.PAUSE_TRADING.value:
                result = self._pause_trading(recovery_record)
                
            elif recovery_action == RecoveryAction.EMERGENCY_STOP.value:
                result = self._emergency_stop(recovery_record)
                
            else:
                recovery_record['status'] = 'failed'
                recovery_record['details']['reason'] = f"Hành động khôi phục không hợp lệ: {recovery_action}"
                self.logger.error(f"Hành động khôi phục không hợp lệ: {recovery_action}")
                result = recovery_record
            
            # Gửi thông báo kết quả khôi phục
            self._send_recovery_notification(result)
            
            return result
            
        except Exception as e:
            # Xử lý lỗi trong quá trình khôi phục
            error_msg = f"Lỗi khi thực hiện khôi phục: {str(e)}"
            error_traceback = traceback.format_exc()
            
            recovery_record['status'] = 'error'
            recovery_record['details']['error'] = error_msg
            recovery_record['details']['traceback'] = error_traceback
            
            self.logger.error(f"Lỗi khi thực hiện khôi phục: {error_msg}\n{error_traceback}")
            
            # Gửi thông báo lỗi
            if self.notification_manager:
                try:
                    self.notification_manager.send_error_notification(
                        error_message=error_msg,
                        error_source="recovery_system",
                        error_trace=error_traceback,
                        critical=True
                    )
                except Exception as notif_error:
                    self.logger.error(f"Không thể gửi thông báo lỗi: {str(notif_error)}")
            
            with self.lock:
                self.recovery_history.append(recovery_record)
                self.recovery_in_progress = False
                
            return recovery_record
            
        finally:
            # Đảm bảo đặt lại trạng thái
            with self.lock:
                self.recovery_in_progress = False
                self.recovery_history.append(recovery_record)
    
    def _retry_component(self, component_id: str, recovery_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thử lại thành phần.
        
        Args:
            component_id: ID thành phần
            recovery_record: Bản ghi khôi phục
            
        Returns:
            Dict chứa kết quả khôi phục
        """
        self.logger.info(f"Thử lại thành phần {component_id}")
        
        # Kiểm tra trạng thái hiện tại
        current_status = self.component_status.get(component_id)
        
        if current_status == ComponentStatus.ERROR.value:
            # Thành phần đang lỗi, cần khởi động lại
            return self._restart_component(component_id, recovery_record)
            
        elif current_status == ComponentStatus.RESTARTING.value:
            # Thành phần đang khởi động lại, chờ hoàn thành
            recovery_record['status'] = 'skipped'
            recovery_record['details']['reason'] = "Thành phần đang trong quá trình khởi động lại"
            self.logger.info(f"Bỏ qua thử lại: Thành phần {component_id} đang trong quá trình khởi động lại")
            return recovery_record
            
        # Ghi nhận làm mới thành phần
        recovery_record['status'] = 'completed'
        recovery_record['success'] = True
        recovery_record['details']['action'] = 'refreshed'
        
        self.logger.info(f"Đã làm mới thành phần {component_id}")
        return recovery_record
    
    def _restart_component(self, component_id: str, recovery_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Khởi động lại thành phần.
        
        Args:
            component_id: ID thành phần
            recovery_record: Bản ghi khôi phục
            
        Returns:
            Dict chứa kết quả khôi phục
        """
        self.logger.info(f"Khởi động lại thành phần {component_id}")
        
        # Kiểm tra số lần khởi động lại
        restart_count = self.restart_counts.get(component_id, 0)
        last_restart_time = self.last_restart_times.get(component_id, 0)
        current_time = time.time()
        
        # Cập nhật số lần khởi động lại
        restart_count += 1
        self.restart_counts[component_id] = restart_count
        self.last_restart_times[component_id] = current_time
        
        # Kiểm tra xem có vượt quá số lần khởi động lại tối đa không
        if restart_count > self.max_restart_attempts:
            # Kiểm tra thời gian cooldown
            if current_time - last_restart_time < self.cooldown_period:
                recovery_record['status'] = 'failed'
                recovery_record['details']['reason'] = f"Đã vượt quá số lần khởi động lại tối đa ({self.max_restart_attempts})"
                recovery_record['details']['restart_count'] = restart_count
                
                self.logger.error(f"Không thể khởi động lại {component_id}: Đã vượt quá số lần khởi động lại tối đa")
                
                # Chuyển sang leo thang
                if restart_count >= self.escalation_threshold:
                    self.logger.warning(f"Leo thang khôi phục cho {component_id} do vượt quá ngưỡng")
                    
                    # Nếu đã vượt quá ngưỡng leo thang, tạm dừng giao dịch
                    pause_record = self._pause_trading(recovery_record.copy())
                    
                    # Cập nhật thông tin
                    recovery_record['details']['escalated'] = True
                    recovery_record['details']['escalation_action'] = "pause_trading"
                    recovery_record['details']['escalation_result'] = pause_record.get('success', False)
                
                return recovery_record
        
        # Cập nhật trạng thái
        self.component_status[component_id] = ComponentStatus.RESTARTING.value
        
        # Lấy thông tin cấu hình thành phần
        component_config = self.config.get("components", {}).get(component_id, {})
        restart_command = component_config.get("restart_command", "")
        script_path = component_config.get("script_path", "")
        
        # Thay thế các biến trong lệnh
        if "{script_path}" in restart_command:
            restart_command = restart_command.replace("{script_path}", script_path)
        
        # Nếu không có lệnh khởi động lại, tạo mặc định
        if not restart_command and script_path:
            if script_path.endswith(".py"):
                restart_command = f"python {script_path}"
            else:
                restart_command = script_path
        
        # Kiểm tra lệnh khởi động lại
        if not restart_command:
            recovery_record['status'] = 'failed'
            recovery_record['details']['reason'] = f"Không có lệnh khởi động lại cho thành phần {component_id}"
            self.logger.error(f"Không thể khởi động lại {component_id}: Không có lệnh khởi động lại")
            
            # Cập nhật trạng thái
            self.component_status[component_id] = ComponentStatus.ERROR.value
            
            return recovery_record
        
        # Thực hiện lệnh khởi động lại
        try:
            # Lưu trạng thái trước khi khởi động lại
            self._save_component_state(component_id)
            
            # Dừng thành phần (nếu đang chạy)
            self._stop_component(component_id)
            
            # Đợi một chút để đảm bảo thành phần đã dừng
            time.sleep(2)
            
            # Khởi động lại thành phần
            self.logger.info(f"Thực hiện lệnh khởi động lại: {restart_command}")
            
            # Thực hiện trong nền
            process = subprocess.Popen(
                restart_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Lấy startup_timeout từ cấu hình
            startup_timeout = component_config.get("startup_timeout", 30)
            
            # Đợi một chút để bắt đầu quá trình
            time.sleep(min(3, startup_timeout / 2))
            
            # Kiểm tra kết quả ban đầu
            if process.poll() is not None:
                # Quá trình đã kết thúc, kiểm tra lỗi
                returncode = process.poll()
                stderr_output = process.stderr.read()
                
                if returncode != 0:
                    recovery_record['status'] = 'failed'
                    recovery_record['details']['reason'] = f"Không thể khởi động lại thành phần: lỗi {returncode}"
                    recovery_record['details']['stderr'] = stderr_output
                    
                    self.logger.error(f"Không thể khởi động lại {component_id}: lỗi {returncode}\n{stderr_output}")
                    
                    # Cập nhật trạng thái
                    self.component_status[component_id] = ComponentStatus.ERROR.value
                    
                    return recovery_record
            
            # Thực hiện kiểm tra sức khỏe
            health_check = component_config.get("health_check", {})
            health_check_type = health_check.get("type")
            health_check_timeout = health_check.get("timeout", 5)
            
            # Đợi đến khi hết thời gian hoặc kiểm tra sức khỏe thành công
            start_time = time.time()
            health_check_success = False
            
            while time.time() - start_time < startup_timeout:
                # Kiểm tra sức khỏe
                if health_check_type == "http":
                    endpoint = health_check.get("endpoint", "")
                    if endpoint:
                        try:
                            import requests
                            response = requests.get(endpoint, timeout=health_check_timeout)
                            if response.status_code == 200:
                                health_check_success = True
                                break
                        except Exception as e:
                            self.logger.debug(f"Kiểm tra sức khỏe HTTP không thành công: {str(e)}")
                
                elif health_check_type == "process":
                    process_name = health_check.get("process_name", "")
                    if process_name:
                        try:
                            import psutil
                            for proc in psutil.process_iter(['pid', 'name']):
                                if process_name in proc.info['name']:
                                    health_check_success = True
                                    break
                        except Exception as e:
                            self.logger.debug(f"Kiểm tra sức khỏe quá trình không thành công: {str(e)}")
                
                # Đợi một chút trước khi kiểm tra lại
                time.sleep(1)
            
            # Kiểm tra kết quả cuối cùng
            if health_check_success:
                # Cập nhật trạng thái
                self.component_status[component_id] = ComponentStatus.RUNNING.value
                
                recovery_record['status'] = 'completed'
                recovery_record['success'] = True
                recovery_record['details']['restart_count'] = restart_count
                
                self.logger.info(f"Đã khởi động lại thành phần {component_id} thành công")
                
                # Khôi phục trạng thái
                self._restore_component_state(component_id)
                
                # Làm mới các thành phần phụ thuộc
                dependent_components = self.components_hierarchy.get(component_id, [])
                if dependent_components:
                    recovery_record['details']['dependent_components'] = dependent_components
                    self.logger.info(f"Làm mới các thành phần phụ thuộc: {dependent_components}")
            else:
                # Kiểm tra sức khỏe không thành công
                recovery_record['status'] = 'failed'
                recovery_record['details']['reason'] = "Thành phần không vượt qua kiểm tra sức khỏe sau khi khởi động lại"
                
                self.logger.error(f"Không thể khởi động lại {component_id}: Thành phần không vượt qua kiểm tra sức khỏe")
                
                # Cập nhật trạng thái
                self.component_status[component_id] = ComponentStatus.ERROR.value
            
            return recovery_record
            
        except Exception as e:
            # Xử lý lỗi trong quá trình khởi động lại
            error_msg = f"Lỗi khi khởi động lại thành phần: {str(e)}"
            error_traceback = traceback.format_exc()
            
            recovery_record['status'] = 'error'
            recovery_record['details']['error'] = error_msg
            recovery_record['details']['traceback'] = error_traceback
            
            self.logger.error(f"Lỗi khi khởi động lại {component_id}: {error_msg}\n{error_traceback}")
            
            # Cập nhật trạng thái
            self.component_status[component_id] = ComponentStatus.ERROR.value
            
            return recovery_record
    
    def _restart_system(self, recovery_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Khởi động lại toàn bộ hệ thống.
        
        Args:
            recovery_record: Bản ghi khôi phục
            
        Returns:
            Dict chứa kết quả khôi phục
        """
        self.logger.warning("Khởi động lại toàn bộ hệ thống")
        
        # Cập nhật trạng thái hệ thống
        self.system_status = "recovering"
        
        # Lưu trạng thái toàn bộ hệ thống
        self._save_system_state()
        
        # Lấy danh sách các thành phần theo thứ tự ưu tiên
        components = self.config.get("components", {})
        component_list = []
        
        for component_id, component_config in components.items():
            priority = component_config.get("priority", 999)
            component_list.append((component_id, priority))
        
        # Sắp xếp theo ưu tiên (số càng nhỏ càng ưu tiên cao)
        component_list.sort(key=lambda x: x[1])
        
        # Dừng các thành phần theo thứ tự ngược lại (ưu tiên thấp trước)
        for component_id, _ in reversed(component_list):
            try:
                self._stop_component(component_id)
            except Exception as e:
                self.logger.error(f"Lỗi khi dừng thành phần {component_id}: {str(e)}")
        
        # Đợi một chút để đảm bảo tất cả đã dừng
        time.sleep(5)
        
        # Khởi động lại các thành phần theo thứ tự ưu tiên
        restart_results = {}
        
        for component_id, _ in component_list:
            try:
                # Tạo bản ghi khôi phục tạm thời
                temp_record = recovery_record.copy()
                temp_record['component_id'] = component_id
                
                # Khởi động lại thành phần
                result = self._restart_component(component_id, temp_record)
                restart_results[component_id] = result.get('success', False)
                
                # Đợi nếu thành phần quan trọng không khởi động được
                if not result.get('success', False) and components.get(component_id, {}).get("critical", False):
                    self.logger.error(f"Thành phần quan trọng {component_id} không thể khởi động lại. Dừng quá trình.")
                    break
                
                # Đợi một chút giữa các lần khởi động lại
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Lỗi khi khởi động lại thành phần {component_id}: {str(e)}")
                restart_results[component_id] = False
        
        # Kiểm tra kết quả
        all_success = all(restart_results.values()) if restart_results else False
        
        if all_success:
            recovery_record['status'] = 'completed'
            recovery_record['success'] = True
            recovery_record['details']['restart_results'] = restart_results
            
            self.logger.info("Đã khởi động lại toàn bộ hệ thống thành công")
            
            # Cập nhật trạng thái hệ thống
            self.system_status = "normal"
        else:
            recovery_record['status'] = 'partial'
            recovery_record['success'] = False
            recovery_record['details']['restart_results'] = restart_results
            recovery_record['details']['reason'] = "Một số thành phần không thể khởi động lại"
            
            self.logger.warning("Khởi động lại hệ thống không hoàn toàn thành công")
            
            # Cập nhật trạng thái hệ thống
            self.system_status = "degraded"
        
        return recovery_record
    
    def _pause_trading(self, recovery_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tạm dừng giao dịch.
        
        Args:
            recovery_record: Bản ghi khôi phục
            
        Returns:
            Dict chứa kết quả khôi phục
        """
        self.logger.warning("Tạm dừng giao dịch")
        
        # Cập nhật trạng thái hệ thống
        self.system_status = "degraded"
        
        try:
            # Xác định thành phần giao dịch
            trade_component = None
            components = self.config.get("components", {})
            
            for component_id, component_config in components.items():
                if "trade" in component_id.lower() or "executor" in component_id.lower():
                    trade_component = component_id
                    break
            
            if not trade_component:
                recovery_record['status'] = 'failed'
                recovery_record['details']['reason'] = "Không tìm thấy thành phần giao dịch"
                self.logger.error("Không thể tạm dừng giao dịch: Không tìm thấy thành phần giao dịch")
                return recovery_record
            
            # Dừng thành phần giao dịch
            self._stop_component(trade_component)
            
            # Cập nhật trạng thái
            self.component_status[trade_component] = ComponentStatus.STOPPED.value
            
            # Gửi thông báo
            self._send_system_notification(
                "Đã tạm dừng giao dịch", 
                f"Hệ thống đã tạm dừng giao dịch do lỗi. Cần can thiệp thủ công.", 
                NotificationPriority.HIGH
            )
            
            recovery_record['status'] = 'completed'
            recovery_record['success'] = True
            recovery_record['details']['paused_component'] = trade_component
            
            self.logger.info(f"Đã tạm dừng giao dịch thành công, thành phần: {trade_component}")
            
            return recovery_record
            
        except Exception as e:
            # Xử lý lỗi trong quá trình tạm dừng
            error_msg = f"Lỗi khi tạm dừng giao dịch: {str(e)}"
            error_traceback = traceback.format_exc()
            
            recovery_record['status'] = 'error'
            recovery_record['details']['error'] = error_msg
            recovery_record['details']['traceback'] = error_traceback
            
            self.logger.error(f"Lỗi khi tạm dừng giao dịch: {error_msg}\n{error_traceback}")
            
            return recovery_record
    
    def _emergency_stop(self, recovery_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dừng khẩn cấp toàn bộ hệ thống.
        
        Args:
            recovery_record: Bản ghi khôi phục
            
        Returns:
            Dict chứa kết quả khôi phục
        """
        self.logger.critical("Dừng khẩn cấp toàn bộ hệ thống")
        
        # Cập nhật trạng thái hệ thống
        self.system_status = "critical"
        
        try:
            # Lưu trạng thái trước khi dừng
            self._save_system_state(emergency=True)
            
            # Lấy danh sách các thành phần
            components = self.config.get("components", {})
            
            # Dừng tất cả các thành phần (ưu tiên thành phần giao dịch trước)
            stop_results = {}
            
            # Dừng thành phần giao dịch trước
            for component_id in components:
                if "trade" in component_id.lower() or "executor" in component_id.lower():
                    try:
                        self._stop_component(component_id)
                        self.component_status[component_id] = ComponentStatus.STOPPED.value
                        stop_results[component_id] = True
                    except Exception as e:
                        self.logger.error(f"Lỗi khi dừng thành phần {component_id}: {str(e)}")
                        stop_results[component_id] = False
            
            # Dừng các thành phần còn lại
            for component_id in components:
                if component_id not in stop_results:
                    try:
                        self._stop_component(component_id)
                        self.component_status[component_id] = ComponentStatus.STOPPED.value
                        stop_results[component_id] = True
                    except Exception as e:
                        self.logger.error(f"Lỗi khi dừng thành phần {component_id}: {str(e)}")
                        stop_results[component_id] = False
            
            # Gửi thông báo khẩn cấp
            self._send_emergency_notification(
                "EMERGENCY STOP", 
                "Hệ thống đã dừng khẩn cấp. Cần can thiệp ngay lập tức.", 
                recovery_record.get('error_details', {})
            )
            
            recovery_record['status'] = 'completed'
            recovery_record['success'] = True
            recovery_record['details']['stop_results'] = stop_results
            
            self.logger.critical("Hệ thống đã dừng khẩn cấp thành công")
            
            return recovery_record
            
        except Exception as e:
            # Xử lý lỗi trong quá trình dừng khẩn cấp
            error_msg = f"Lỗi khi dừng khẩn cấp: {str(e)}"
            error_traceback = traceback.format_exc()
            
            recovery_record['status'] = 'error'
            recovery_record['details']['error'] = error_msg
            recovery_record['details']['traceback'] = error_traceback
            
            self.logger.critical(f"Lỗi khi dừng khẩn cấp: {error_msg}\n{error_traceback}")
            
            # Thử gửi thông báo khẩn cấp
            try:
                self._send_emergency_notification(
                    "EMERGENCY STOP FAILED", 
                    f"Dừng khẩn cấp không thành công: {error_msg}. Cần can thiệp thủ công ngay lập tức.", 
                    {"error": error_msg, "traceback": error_traceback}
                )
            except Exception:
                pass
                
            return recovery_record
    
    def _stop_component(self, component_id: str) -> bool:
        """
        Dừng thành phần.
        
        Args:
            component_id: ID thành phần
            
        Returns:
            True nếu thành công, False nếu không
        """
        self.logger.info(f"Dừng thành phần {component_id}")
        
        # Lấy thông tin cấu hình thành phần
        component_config = self.config.get("components", {}).get(component_id, {})
        
        try:
            # Kiểm tra loại thành phần
            health_check = component_config.get("health_check", {})
            health_check_type = health_check.get("type")
            
            if health_check_type == "process":
                # Dừng theo tên tiến trình
                process_name = health_check.get("process_name", "")
                
                if process_name:
                    import psutil
                    for proc in psutil.process_iter(['pid', 'name']):
                        if process_name in proc.info['name']:
                            # Gửi tín hiệu dừng
                            pid = proc.info['pid']
                            os.kill(pid, signal.SIGTERM)
                            
                            # Đợi tiến trình dừng
                            try:
                                # Đợi tối đa 5 giây
                                proc = psutil.Process(pid)
                                proc.wait(timeout=5)
                            except psutil.NoSuchProcess:
                                # Tiến trình đã dừng
                                pass
                            except psutil.TimeoutExpired:
                                # Tiến trình không dừng, sử dụng SIGKILL
                                self.logger.warning(f"Tiến trình {process_name} không phản hồi, sử dụng SIGKILL")
                                os.kill(pid, signal.SIGKILL)
                            
                            self.logger.info(f"Đã dừng tiến trình {process_name} (PID: {pid})")
                            
            elif health_check_type == "http":
                # Gửi yêu cầu dừng qua HTTP
                endpoint = health_check.get("endpoint", "")
                if endpoint:
                    # Thay đổi endpoint để gửi yêu cầu dừng
                    # Ví dụ: chuyển từ /health thành /shutdown
                    shutdown_endpoint = endpoint.replace("/health", "/shutdown")
                    
                    if shutdown_endpoint != endpoint:
                        try:
                            import requests
                            response = requests.post(shutdown_endpoint, timeout=5)
                            
                            if response.status_code == 200:
                                self.logger.info(f"Đã gửi yêu cầu dừng đến {shutdown_endpoint}")
                            else:
                                self.logger.warning(f"Không thể gửi yêu cầu dừng đến {shutdown_endpoint}: {response.status_code}")
                        except Exception as e:
                            self.logger.error(f"Lỗi khi gửi yêu cầu dừng: {str(e)}")
            
            # Cập nhật trạng thái
            self.component_status[component_id] = ComponentStatus.STOPPED.value
            
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi dừng thành phần {component_id}: {str(e)}")
            return False
    
    def _save_component_state(self, component_id: str) -> bool:
        """
        Lưu trạng thái thành phần.
        
        Args:
            component_id: ID thành phần
            
        Returns:
            True nếu thành công, False nếu không
        """
        try:
            # Tạo tên file
            file_path = os.path.join(self.state_dir, f"{component_id}_state.json")
            
            # Lấy trạng thái
            state = {
                'timestamp': datetime.now().isoformat(),
                'component_id': component_id,
                'status': self.component_status.get(component_id),
                'restart_count': self.restart_counts.get(component_id, 0),
                'last_restart_time': self.last_restart_times.get(component_id, 0)
            }
            
            # Lưu vào file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4)
                
            self.logger.debug(f"Đã lưu trạng thái thành phần {component_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu trạng thái thành phần {component_id}: {str(e)}")
            return False
    
    def _restore_component_state(self, component_id: str) -> bool:
        """
        Khôi phục trạng thái thành phần.
        
        Args:
            component_id: ID thành phần
            
        Returns:
            True nếu thành công, False nếu không
        """
        try:
            # Tạo tên file
            file_path = os.path.join(self.state_dir, f"{component_id}_state.json")
            
            if not os.path.exists(file_path):
                self.logger.warning(f"Không tìm thấy file trạng thái cho thành phần {component_id}")
                return False
            
            # Đọc từ file
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Cập nhật trạng thái
            timestamp = state.get('timestamp')
            saved_status = state.get('status')
            
            # Không cập nhật các trường đếm để giữ giá trị hiện tại
            
            self.logger.debug(f"Đã khôi phục trạng thái thành phần {component_id} từ {timestamp}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi khôi phục trạng thái thành phần {component_id}: {str(e)}")
            return False
    
    def _save_system_state(self, emergency: bool = False) -> bool:
        """
        Lưu trạng thái toàn bộ hệ thống.
        
        Args:
            emergency: Đánh dấu trạng thái khẩn cấp
            
        Returns:
            True nếu thành công, False nếu không
        """
        try:
            # Tạo tên file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"system_state_{timestamp}.json"
            
            if emergency:
                filename = f"emergency_state_{timestamp}.json"
            
            file_path = os.path.join(self.state_dir, filename)
            
            # Lấy trạng thái
            state = {
                'timestamp': datetime.now().isoformat(),
                'system_status': self.system_status,
                'emergency': emergency,
                'components': {}
            }
            
            # Lưu trạng thái từng thành phần
            for component_id, status in self.component_status.items():
                state['components'][component_id] = {
                    'status': status,
                    'restart_count': self.restart_counts.get(component_id, 0),
                    'last_restart_time': self.last_restart_times.get(component_id, 0)
                }
            
            # Lưu vào file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4)
                
            self.logger.debug(f"Đã lưu trạng thái hệ thống vào {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu trạng thái hệ thống: {str(e)}")
            return False
    
    def _send_recovery_notification(self, recovery_record: Dict[str, Any]) -> None:
        """
        Gửi thông báo về kết quả khôi phục.
        
        Args:
            recovery_record: Bản ghi khôi phục
        """
        if not self.notification_manager:
            return
            
        # Xác định mức độ ưu tiên
        priority = NotificationPriority.NORMAL
        
        if recovery_record['status'] in ['error', 'failed']:
            priority = NotificationPriority.HIGH
        
        # Tạo tiêu đề
        component_id = recovery_record.get('component_id', 'System')
        action = recovery_record.get('action', 'unknown')
        status = "thành công" if recovery_record.get('success', False) else "thất bại"
        
        subject = f"Khôi phục {component_id}: {action} {status}"
        
        # Tạo thông điệp
        message = f"ID: {recovery_record['id']}\n"
        message += f"Hành động: {action}\n"
        message += f"Thành phần: {component_id}\n"
        message += f"Trạng thái: {recovery_record['status']}\n"
        message += f"Kết quả: {status}\n"
        message += f"Thời gian: {recovery_record['timestamp']}\n"
        
        # Thêm chi tiết nếu có
        if 'details' in recovery_record and recovery_record['details']:
            message += "\nChi tiết:\n"
            for key, value in recovery_record['details'].items():
                message += f"- {key}: {value}\n"
        
        # Gửi thông báo
        try:
            if recovery_record.get('success', False):
                self.notification_manager.send_system_notification(
                    component=component_id,
                    status=f"recovery_{recovery_record['status']}",
                    details=message,
                    priority=priority
                )
            else:
                self.notification_manager.send_error_notification(
                    error_message=f"Khôi phục {action} {status}",
                    error_source=f"recovery_system:{component_id}",
                    error_trace=json.dumps(recovery_record.get('details', {}), indent=2),
                    critical=priority == NotificationPriority.HIGH
                )
        except Exception as e:
            self.logger.error(f"Không thể gửi thông báo khôi phục: {str(e)}")
    
    def _send_system_notification(self, subject: str, message: str, priority: NotificationPriority) -> None:
        """
        Gửi thông báo hệ thống.
        
        Args:
            subject: Tiêu đề
            message: Nội dung
            priority: Mức độ ưu tiên
        """
        if not self.notification_manager:
            return
            
        try:
            self.notification_manager.send_system_notification(
                component="recovery_system",
                status="notification",
                details=message,
                priority=priority
            )
        except Exception as e:
            self.logger.error(f"Không thể gửi thông báo hệ thống: {str(e)}")
    
    def _send_emergency_notification(self, subject: str, message: str, details: Dict[str, Any]) -> None:
        """
        Gửi thông báo khẩn cấp.
        
        Args:
            subject: Tiêu đề
            message: Nội dung
            details: Chi tiết bổ sung
        """
        if not self.notification_manager:
            return
            
        try:
            # Gửi thông báo lỗi khẩn cấp
            self.notification_manager.send_error_notification(
                error_message=message,
                error_source="recovery_system:emergency",
                error_trace=json.dumps(details, indent=2) if details else None,
                critical=True,
                immediate=True
            )
            
            # Gửi email trực tiếp đến các liên hệ khẩn cấp nếu có
            emergency_contacts = self.config.get("recovery", {}).get("emergency_contacts", [])
            
            if emergency_contacts and hasattr(self.notification_manager, 'notifiers') and 'email' in self.notification_manager.notifiers:
                try:
                    email_notifier = self.notification_manager.notifiers['email']
                    
                    # Gửi email đến các liên hệ khẩn cấp
                    email_notifier.send_email(
                        subject=f"[EMERGENCY] {subject}",
                        message=message,
                        recipients=emergency_contacts,
                        notification_type="error"
                    )
                except Exception as email_error:
                    self.logger.error(f"Không thể gửi email khẩn cấp: {str(email_error)}")
            
        except Exception as e:
            self.logger.error(f"Không thể gửi thông báo khẩn cấp: {str(e)}")
    
    def check_component_status(self, component_id: str) -> str:
        """
        Kiểm tra trạng thái của thành phần.
        
        Args:
            component_id: ID thành phần
            
        Returns:
            Trạng thái thành phần
        """
        # Kiểm tra xem có trong danh sách không
        if component_id not in self.component_status:
            return ComponentStatus.UNKNOWN.value
        
        # Lấy thông tin cấu hình thành phần
        component_config = self.config.get("components", {}).get(component_id, {})
        health_check = component_config.get("health_check", {})
        health_check_type = health_check.get("type")
        
        # Nếu có kiểm tra sức khỏe, thực hiện kiểm tra
        if health_check_type == "http":
            endpoint = health_check.get("endpoint", "")
            timeout = health_check.get("timeout", 5)
            
            if endpoint:
                try:
                    import requests
                    response = requests.get(endpoint, timeout=timeout)
                    
                    if response.status_code == 200:
                        self.component_status[component_id] = ComponentStatus.RUNNING.value
                    else:
                        self.component_status[component_id] = ComponentStatus.ERROR.value
                        
                except Exception:
                    self.component_status[component_id] = ComponentStatus.ERROR.value
        
        elif health_check_type == "process":
            process_name = health_check.get("process_name", "")
            
            if process_name:
                try:
                    import psutil
                    process_found = False
                    
                    for proc in psutil.process_iter(['pid', 'name']):
                        if process_name in proc.info['name']:
                            process_found = True
                            break
                    
                    if process_found:
                        self.component_status[component_id] = ComponentStatus.RUNNING.value
                    else:
                        self.component_status[component_id] = ComponentStatus.STOPPED.value
                        
                except Exception:
                    pass
        
        return self.component_status.get(component_id, ComponentStatus.UNKNOWN.value)
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Lấy thông tin sức khỏe toàn bộ hệ thống.
        
        Returns:
            Dict chứa thông tin sức khỏe
        """
        # Cập nhật trạng thái các thành phần
        components_status = {}
        
        for component_id in self.component_status:
            components_status[component_id] = self.check_component_status(component_id)
        
        # Xác định trạng thái hệ thống
        if all(status == ComponentStatus.RUNNING.value for status in components_status.values()):
            system_status = "healthy"
        elif any(status == ComponentStatus.ERROR.value for status in components_status.values()):
            system_status = "error"
        elif any(status == ComponentStatus.STOPPED.value for status in components_status.values()):
            system_status = "degraded"
        else:
            system_status = "unknown"
        
        # Thông tin khôi phục gần đây
        recent_recoveries = self.recovery_history[-10:] if self.recovery_history else []
        
        # Tạo báo cáo sức khỏe
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'system_status': system_status,
            'components_status': components_status,
            'recovery_in_progress': self.recovery_in_progress,
            'recent_recoveries': recent_recoveries
        }
        
        return health_report