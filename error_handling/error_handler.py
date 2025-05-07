import logging
import traceback
import time
import os
import json
from datetime import datetime
from pathlib import Path
import sys

# Thư mục gốc
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{project_root}/logs/error_handler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("error_handler")

class ErrorHandler:
    """Lớp xử lý lỗi chung cho hệ thống"""
    
    def __init__(self, config=None):
        """Khởi tạo error handler
        
        Args:
            config: Cấu hình hệ thống
        """
        self.config = config
        self.error_log_dir = os.path.join(project_root, "logs", "errors")
        os.makedirs(self.error_log_dir, exist_ok=True)
        
        # Lưu trữ thông tin lỗi
        self.error_counts = {}
        self.last_error_time = {}
        self.error_thresholds = {
            "API_ERROR": 5,      # Lỗi kết nối API
            "DATA_ERROR": 3,     # Lỗi dữ liệu
            "MODEL_ERROR": 2,    # Lỗi mô hình
            "SYSTEM_ERROR": 1,   # Lỗi hệ thống
        }
        
        logger.info("Error handler initialized")
    
    def handle_error(self, error, error_type=None, context=None, retry_func=None, retry_args=None, max_retries=3, retry_delay=5):
        """Xử lý lỗi chung
        
        Args:
            error: Exception đã xảy ra
            error_type: Loại lỗi (API_ERROR, DATA_ERROR, MODEL_ERROR, SYSTEM_ERROR)
            context: Ngữ cảnh lỗi xảy ra
            retry_func: Hàm thử lại nếu cần
            retry_args: Tham số cho hàm thử lại
            max_retries: Số lần thử lại tối đa
            retry_delay: Thời gian chờ giữa các lần thử lại (giây)
            
        Returns:
            tuple: (success, result) - success là bool, result là kết quả hoặc lỗi
        """
        if error_type is None:
            error_type = "UNKNOWN_ERROR"
        
        # Ghi log lỗi
        error_msg = f"{error_type}: {str(error)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # Lưu thông tin lỗi
        self._log_error(error, error_type, context)
        
        # Cập nhật thống kê lỗi
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        self.last_error_time[error_type] = datetime.now()
        
        # Kiểm tra nếu vượt ngưỡng lỗi
        if error_type in self.error_thresholds and self.error_counts[error_type] >= self.error_thresholds[error_type]:
            logger.critical(f"Error threshold reached for {error_type}. Taking emergency action.")
            self._take_emergency_action(error_type)
        
        # Thử lại nếu cần
        if retry_func is not None:
            result = self._retry_operation(retry_func, retry_args, max_retries, retry_delay, error_type)
            return result
        
        return False, error
    
    def _log_error(self, error, error_type, context):
        """Ghi log lỗi chi tiết vào file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_id = f"{error_type}_{timestamp}"
        error_file = os.path.join(self.error_log_dir, f"{error_id}.json")
        
        error_data = {
            "error_id": error_id,
            "error_type": error_type,
            "timestamp": timestamp,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        with open(error_file, 'w') as f:
            json.dump(error_data, f, indent=4)
        
        logger.info(f"Error details logged to {error_file}")
    
    def _retry_operation(self, func, args, max_retries, retry_delay, error_type):
        """Thử lại thao tác bị lỗi
        
        Args:
            func: Hàm cần thử lại
            args: Tham số cho hàm
            max_retries: Số lần thử lại tối đa
            retry_delay: Thời gian chờ giữa các lần thử lại (giây)
            error_type: Loại lỗi
            
        Returns:
            tuple: (success, result) - success là bool, result là kết quả hoặc lỗi
        """
        retry_count = 0
        while retry_count < max_retries:
            retry_count += 1
            logger.info(f"Retrying operation ({retry_count}/{max_retries}) after {retry_delay}s delay...")
            
            try:
                time.sleep(retry_delay)
                result = func(**(args or {}))
                logger.info(f"Retry successful on attempt {retry_count}")
                
                # Reset error count for this type
                self.error_counts[error_type] = 0
                
                return True, result
            except Exception as e:
                logger.warning(f"Retry attempt {retry_count} failed: {str(e)}")
                last_error = e
        
        logger.error(f"All {max_retries} retry attempts failed")
        return False, last_error
    
    def _take_emergency_action(self, error_type):
        """Thực hiện hành động khẩn cấp khi vượt ngưỡng lỗi
        
        Args:
            error_type: Loại lỗi đã vượt ngưỡng
        """
        if error_type == "API_ERROR":
            logger.critical("Multiple API errors detected. Pausing trading activities.")
            # TODO: Implement trading pause logic
            
        elif error_type == "DATA_ERROR":
            logger.critical("Multiple data errors detected. Switching to backup data source.")
            # TODO: Implement backup data source logic
            
        elif error_type == "MODEL_ERROR":
            logger.critical("Multiple model errors detected. Reverting to fallback strategy.")
            # TODO: Implement fallback strategy
            
        elif error_type == "SYSTEM_ERROR":
            logger.critical("Critical system error. Initiating emergency shutdown.")
            # TODO: Implement emergency shutdown
        
        # Send alert notification
        self._send_alert(f"Emergency action taken for {error_type}")
    
    def _send_alert(self, message):
        """Gửi thông báo cảnh báo
        
        Args:
            message: Nội dung cảnh báo
        """
        logger.critical(f"ALERT: {message}")
        # TODO: Implement actual alert mechanism (email, Telegram, etc.)
        
    def reset_error_counts(self, error_type=None):
        """Reset thống kê lỗi
        
        Args:
            error_type: Loại lỗi cần reset, hoặc None để reset tất cả
        """
        if error_type is None:
            self.error_counts = {}
            self.last_error_time = {}
        else:
            if error_type in self.error_counts:
                self.error_counts[error_type] = 0
                self.last_error_time[error_type] = None
                
        logger.info(f"Reset error counts for {error_type or 'all types'}")

# Helper decorator cho xử lý lỗi
def handle_errors(error_type=None, max_retries=3, retry_delay=5):
    """Decorator để xử lý lỗi tự động cho các hàm
    
    Args:
        error_type: Loại lỗi
        max_retries: Số lần thử lại tối đa
        retry_delay: Thời gian chờ giữa các lần thử lại (giây)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            handler = ErrorHandler()
            context = {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            }
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                retry_args = {
                    "args": args,
                    "kwargs": kwargs
                }
                success, result = handler.handle_error(
                    e, 
                    error_type=error_type, 
                    context=context,
                    retry_func=func,
                    retry_args=retry_args,
                    max_retries=max_retries,
                    retry_delay=retry_delay
                )
                
                if success:
                    return result
                else:
                    raise result
                    
        return wrapper
    return decorator

# Ví dụ sử dụng
if __name__ == "__main__":
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs("error_handling", exist_ok=True)
    
    # Tạo Error Handler
    handler = ErrorHandler()
    
    # Ví dụ hàm với decorator
    @handle_errors(error_type="DATA_ERROR", max_retries=2)
    def process_data(data):
        if data is None:
            raise ValueError("Data cannot be None")
        return data * 2
    
    # Test xử lý lỗi
    try:
        result = process_data(None)
    except Exception as e:
        print(f"Expected error was handled: {e}")
    
    # Test thành công
    result = process_data(5)
    print(f"Process successful, result: {result}")