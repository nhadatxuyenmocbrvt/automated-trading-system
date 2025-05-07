import os
import sys
import time
import logging
import json
import pickle
import threading
import traceback
import psutil
from datetime import datetime, timedelta
from pathlib import Path

# Thêm thư mục gốc vào path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{project_root}/logs/recovery_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("recovery_system")

from error_handling.error_handler import ErrorHandler

class RecoverySystem:
    """Hệ thống phục hồi tự động cho các lỗi nghiêm trọng"""
    
    def __init__(self, config=None):
        """Khởi tạo hệ thống phục hồi
        
        Args:
            config: Cấu hình hệ thống
        """
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.recovery_dir = os.path.join(project_root, "recovery")
        self.checkpoint_dir = os.path.join(self.recovery_dir, "checkpoints")
        self.state_file = os.path.join(self.recovery_dir, "system_state.json")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.recovery_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Trạng thái hệ thống
        self.system_state = self._load_system_state()
        
        # Giám sát
        self.monitoring_active = False
        self.monitor_thread = None
        self.last_health_check = datetime.now()
        
        logger.info("Recovery system initialized")
    
    def _load_system_state(self):
        """Tải trạng thái hệ thống từ file
        
        Returns:
            dict: Trạng thái hệ thống
        """
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading system state: {e}")
        
        # Trạng thái mặc định
        return {
            "last_recovery": None,
            "recovery_count": 0,
            "last_checkpoint": None,
            "components_status": {
                "data_collector": "inactive",
                "trading_system": "inactive",
                "monitoring": "inactive"
            },
            "active_components": []
        }
    
    def _save_system_state(self):
        """Lưu trạng thái hệ thống vào file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.system_state, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
    
    def create_checkpoint(self, data, name):
        """Tạo điểm khôi phục cho dữ liệu
        
        Args:
            data: Dữ liệu cần lưu
            name: Tên điểm khôi phục
        
        Returns:
            str: Đường dẫn đến file checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{name}_{timestamp}.pkl")
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.system_state["last_checkpoint"] = checkpoint_file
            self._save_system_state()
            
            logger.info(f"Checkpoint created: {checkpoint_file}")
            return checkpoint_file
        except Exception as e:
            logger.error(f"Error creating checkpoint: {e}")
            self.error_handler.handle_error(e, error_type="SYSTEM_ERROR", context={"action": "create_checkpoint"})
            return None
    
    def load_checkpoint(self, checkpoint_file=None):
        """Tải dữ liệu từ điểm khôi phục
        
        Args:
            checkpoint_file: Đường dẫn đến file checkpoint, hoặc None để tải checkpoint cuối cùng
        
        Returns:
            object: Dữ liệu đã lưu
        """
        if checkpoint_file is None:
            checkpoint_file = self.system_state.get("last_checkpoint")
            
        if checkpoint_file is None or not os.path.exists(checkpoint_file):
            logger.error(f"Checkpoint file not found: {checkpoint_file}")
            return None
        
        try:
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Checkpoint loaded: {checkpoint_file}")
            return data
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            self.error_handler.handle_error(e, error_type="SYSTEM_ERROR", context={"action": "load_checkpoint"})
            return None
    
    def recover_component(self, component_name):
        """Khôi phục một thành phần của hệ thống
        
        Args:
            component_name: Tên thành phần cần khôi phục
        
        Returns:
            bool: True nếu khôi phục thành công, False nếu thất bại
        """
        logger.info(f"Attempting to recover component: {component_name}")
        
        # Tìm checkpoint gần nhất cho thành phần
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith(component_name)]
        checkpoints.sort(reverse=True)  # Sắp xếp theo thời gian giảm dần
        
        if not checkpoints:
            logger.error(f"No checkpoints found for component: {component_name}")
            return False
        
        latest_checkpoint = os.path.join(self.checkpoint_dir, checkpoints[0])
        
        try:
            # Tải dữ liệu từ checkpoint
            component_data = self.load_checkpoint(latest_checkpoint)
            if component_data is None:
                return False
            
            # Khởi động lại thành phần
            success = self._restart_component(component_name, component_data)
            
            if success:
                # Cập nhật trạng thái
                self.system_state["components_status"][component_name] = "active"
                self.system_state["last_recovery"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.system_state["recovery_count"] += 1
                self._save_system_state()
                
                logger.info(f"Component {component_name} recovered successfully")
                return True
            else:
                logger.error(f"Failed to recover component: {component_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error during recovery of {component_name}: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _restart_component(self, component_name, component_data):
        """Khởi động lại một thành phần với dữ liệu
        
        Args:
            component_name: Tên thành phần
            component_data: Dữ liệu thành phần
        
        Returns:
            bool: True nếu khởi động thành công, False nếu thất bại
        """
        # Thực hiện khởi động lại dựa trên loại thành phần
        try:
            if component_name == "data_collector":
                # Logic khởi động lại data collector
                logger.info("Restarting data collector component")
                # TODO: Implement restart logic
                return True
                
            elif component_name == "trading_system":
                # Logic khởi động lại trading system
                logger.info("Restarting trading system component")
                # TODO: Implement restart logic
                return True
                
            elif component_name == "agent":
                # Logic khởi động lại agent
                logger.info("Restarting agent component")
                # TODO: Implement restart logic
                return True
                
            else:
                logger.warning(f"Unknown component: {component_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error restarting component {component_name}: {e}")
            return False
    
    def recover_all_components(self):
        """Khôi phục tất cả các thành phần hệ thống
        
        Returns:
            dict: Kết quả khôi phục cho từng thành phần
        """
        results = {}
        
        for component in self.system_state["active_components"]:
            logger.info(f"Recovering component: {component}")
            success = self.recover_component(component)
            results[component] = success
        
        return results
    
    def start_health_monitoring(self, interval=60):
        """Bắt đầu giám sát sức khỏe hệ thống
        
        Args:
            interval: Thời gian giữa các lần kiểm tra (giây)
        """
        if self.monitoring_active:
            logger.warning("Health monitoring is already active")
            return
        
        self.monitoring_active = True
        self.system_state["components_status"]["monitoring"] = "active"
        self._save_system_state()
        
        logger.info(f"Starting health monitoring with interval: {interval}s")
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self.check_system_health()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in health monitoring: {e}")
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_health_monitoring(self):
        """Dừng giám sát sức khỏe hệ thống"""
        if not self.monitoring_active:
            logger.warning("Health monitoring is not active")
            return
        
        logger.info("Stopping health monitoring")
        self.monitoring_active = False
        self.system_state["components_status"]["monitoring"] = "inactive"
        self._save_system_state()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            self.monitor_thread = None
    
    def check_system_health(self):
        """Kiểm tra sức khỏe hệ thống và thực hiện khôi phục nếu cần
        
        Returns:
            dict: Trạng thái sức khỏe hệ thống
        """
        self.last_health_check = datetime.now()
        health_status = {}
        
        # Kiểm tra sử dụng CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        health_status["cpu_usage"] = cpu_percent
        
        # Kiểm tra sử dụng bộ nhớ
        memory = psutil.virtual_memory()
        health_status["memory_usage"] = memory.percent
        
        # Kiểm tra không gian đĩa
        disk = psutil.disk_usage('/')
        health_status["disk_usage"] = disk.percent
        
        # Kiểm tra các tiến trình hệ thống
        health_status["components"] = {}
        
        # Kiểm tra từng thành phần
        for component in self.system_state["active_components"]:
            component_status = self._check_component_health(component)
            health_status["components"][component] = component_status
            
            # Khôi phục thành phần nếu cần
            if component_status["status"] == "unhealthy":
                logger.warning(f"Component {component} is unhealthy. Attempting recovery.")
                recovery_success = self.recover_component(component)
                health_status["components"][component]["recovery_attempted"] = True
                health_status["components"][component]["recovery_success"] = recovery_success
        
        # Lưu kết quả kiểm tra
        health_log_file = os.path.join(project_root, "logs", "health_checks.json")
        
        try:
            # Đọc log hiện tại nếu có
            if os.path.exists(health_log_file):
                with open(health_log_file, 'r') as f:
                    health_logs = json.load(f)
            else:
                health_logs = []
            
            # Thêm kết quả mới
            health_logs.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "health_status": health_status
            })
            
            # Giới hạn số lượng log
            if len(health_logs) > 100:
                health_logs = health_logs[-100:]
            
            # Lưu log
            with open(health_log_file, 'w') as f:
                json.dump(health_logs, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error saving health check log: {e}")
        
        return health_status
    
    def _check_component_health(self, component_name):
        """Kiểm tra sức khỏe của một thành phần
        
        Args:
            component_name: Tên thành phần cần kiểm tra
            
        Returns:
            dict: Trạng thái sức khỏe thành phần
        """
        # TODO: Thực hiện kiểm tra dựa trên loại thành phần
        
        # Logic mặc định
        component_status = {
            "name": component_name,
            "status": "healthy",
            "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "details": {}
        }
        
        # Ví dụ kiểm tra giả lập - thay thế bằng kiểm tra thực tế
        if component_name == "data_collector":
            # Kiểm tra thời gian dữ liệu cuối cùng
            last_data_time = self._get_last_data_time()
            if last_data_time and datetime.now() - last_data_time > timedelta(minutes=15):
                component_status["status"] = "unhealthy"
                component_status["details"]["reason"] = "No recent data collected"
        
        elif component_name == "trading_system":
            # Kiểm tra giao dịch gần đây
            last_trade_time = self._get_last_trade_time()
            if last_trade_time and datetime.now() - last_trade_time > timedelta(hours=1):
                component_status["status"] = "warning"
                component_status["details"]["reason"] = "No recent trades"
        
        return component_status
    
    def _get_last_data_time(self):
        """Lấy thời gian dữ liệu cuối cùng được thu thập
        
        Returns:
            datetime: Thời gian dữ liệu cuối cùng, hoặc None nếu không có
        """
        # TODO: Implement real logic
        # Giả lập logic - thay thế bằng logic thực
        return datetime.now() - timedelta(minutes=5)
    
    def _get_last_trade_time(self):
        """Lấy thời gian giao dịch cuối cùng
        
        Returns:
            datetime: Thời gian giao dịch cuối cùng, hoặc None nếu không có
        """
        # TODO: Implement real logic
        # Giả lập logic - thay thế bằng logic thực
        return datetime.now() - timedelta(minutes=30)
    
    def register_component(self, component_name):
        """Đăng ký một thành phần với hệ thống phục hồi
        
        Args:
            component_name: Tên thành phần
        """
        if component_name not in self.system_state["active_components"]:
            self.system_state["active_components"].append(component_name)
            
        if component_name not in self.system_state["components_status"]:
            self.system_state["components_status"][component_name] = "active"
            
        self._save_system_state()
        logger.info(f"Component registered: {component_name}")
    
    def unregister_component(self, component_name):
        """Hủy đăng ký một thành phần
        
        Args:
            component_name: Tên thành phần
        """
        if component_name in self.system_state["active_components"]:
            self.system_state["active_components"].remove(component_name)
            
        if component_name in self.system_state["components_status"]:
            self.system_state["components_status"][component_name] = "inactive"
            
        self._save_system_state()
        logger.info(f"Component unregistered: {component_name}")

if __name__ == "__main__":
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs("error_handling", exist_ok=True)
    
    # Tạo Recovery System
    recovery = RecoverySystem()
    
    # Đăng ký các thành phần
    recovery.register_component("data_collector")
    recovery.register_component("trading_system")
    
    # Tạo checkpoint giả lập
    data = {"key": "value", "timestamp": datetime.now()}
    recovery.create_checkpoint(data, "data_collector")
    
    # Bắt đầu giám sát
    recovery.start_health_monitoring(interval=10)
    
    # Demo
    try:
        print("Recovery system running. Press Ctrl+C to exit.")
        time.sleep(30)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        recovery.stop_health_monitoring()
        print("Recovery system stopped.")