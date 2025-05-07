"""
Xử lý lỗi tự động.
File này cung cấp lớp ErrorHandler để phát hiện, phân loại và xử lý các lỗi
phát sinh trong hệ thống, hỗ trợ việc khôi phục tự động và báo cáo lỗi.
"""

import os
import time
import traceback
import logging
import threading
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from config.constants import ErrorCode
from real_time_inference.system_monitor.notification_manager import NotificationManager, NotificationPriority, NotificationType

class ErrorSeverity(Enum):
    """Mức độ nghiêm trọng của lỗi."""
    LOW = 1        # Lỗi nhẹ, không ảnh hưởng đến hoạt động
    MEDIUM = 2     # Lỗi vừa, có thể ảnh hưởng một phần chức năng
    HIGH = 3       # Lỗi nặng, ảnh hưởng đến chức năng quan trọng
    CRITICAL = 4   # Lỗi nghiêm trọng, hệ thống không thể hoạt động

class ErrorCategory(Enum):
    """Phân loại lỗi."""
    NETWORK = "network"               # Lỗi kết nối mạng
    API = "api"                       # Lỗi API
    EXCHANGE = "exchange"             # Lỗi sàn giao dịch
    DATA = "data"                     # Lỗi dữ liệu
    MODEL = "model"                   # Lỗi mô hình
    SYSTEM = "system"                 # Lỗi hệ thống
    DATABASE = "database"             # Lỗi cơ sở dữ liệu
    AUTHENTICATION = "authentication" # Lỗi xác thực
    RATE_LIMIT = "rate_limit"         # Lỗi giới hạn tốc độ
    UNKNOWN = "unknown"               # Lỗi không xác định

class RecoveryAction(Enum):
    """Hành động khôi phục."""
    NONE = "none"                   # Không cần hành động
    RETRY = "retry"                 # Thử lại
    RESTART_COMPONENT = "restart"   # Khởi động lại thành phần
    RESTART_SYSTEM = "restart_all"  # Khởi động lại toàn bộ hệ thống
    PAUSE_TRADING = "pause"         # Tạm dừng giao dịch
    EMERGENCY_STOP = "stop"         # Dừng khẩn cấp toàn bộ

class ErrorHandler:
    """
    Lớp xử lý lỗi tự động.
    Phát hiện, phân loại và xử lý các lỗi phát sinh trong hệ thống.
    """
    
    def __init__(
        self,
        notification_manager: Optional[NotificationManager] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
        error_log_dir: Optional[str] = None,
        enable_auto_recovery: bool = True,
        error_callback: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo ErrorHandler.
        
        Args:
            notification_manager: Quản lý thông báo
            max_retries: Số lần thử lại tối đa
            retry_delay: Thời gian chờ giữa các lần thử lại (giây)
            error_log_dir: Thư mục lưu log lỗi
            enable_auto_recovery: Kích hoạt tự động khôi phục
            error_callback: Hàm callback khi có lỗi
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("error_handler")
        
        # Lấy cấu hình hệ thống
        self.config = get_system_config().get_all()
        
        # Quản lý thông báo
        self.notification_manager = notification_manager
        
        # Thiết lập các tham số
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_auto_recovery = enable_auto_recovery
        self.error_callback = error_callback
        
        # Thiết lập thư mục log
        if error_log_dir is None:
            base_log_dir = os.path.join(self.config.get("log_dir", "logs"), "errors")
            self.error_log_dir = base_log_dir
        else:
            self.error_log_dir = error_log_dir
            
        # Đảm bảo thư mục tồn tại
        os.makedirs(self.error_log_dir, exist_ok=True)
        
        # Lưu trữ lỗi
        self.error_history = []
        self.error_counts = {}  # Đếm số lỗi theo loại
        self.active_errors = {}  # Lỗi đang hoạt động
        
        # Thiết lập lock thread
        self.lock = threading.Lock()
        
        self.logger.info("Đã khởi tạo ErrorHandler")
    
    def handle_error(
        self,
        error: Exception,
        component: str,
        error_code: Optional[Union[int, ErrorCode]] = None,
        category: Optional[Union[str, ErrorCategory]] = None,
        severity: Optional[Union[int, ErrorSeverity]] = None,
        context: Optional[Dict[str, Any]] = None,
        retry_action: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Xử lý lỗi.
        
        Args:
            error: Exception
            component: Tên thành phần phát sinh lỗi
            error_code: Mã lỗi
            category: Phân loại lỗi
            severity: Mức độ nghiêm trọng
            context: Thông tin ngữ cảnh
            retry_action: Hàm thử lại
            
        Returns:
            Dict chứa thông tin xử lý lỗi
        """
        # Lấy thông tin lỗi
        error_message = str(error)
        error_traceback = traceback.format_exc()
        error_time = datetime.now()
        
        # Chuyển đổi enum thành giá trị nếu cần
        if isinstance(category, ErrorCategory):
            category = category.value
        elif category is None:
            category = ErrorCategory.UNKNOWN.value
            
        if isinstance(severity, ErrorSeverity):
            severity = severity.value
        elif severity is None:
            severity = ErrorSeverity.MEDIUM.value
            
        if isinstance(error_code, ErrorCode):
            error_code = error_code.value
        elif error_code is None:
            error_code = 0
        
        # Phân loại lỗi nếu chưa được chỉ định
        if category == ErrorCategory.UNKNOWN.value:
            category = self._classify_error(error, error_message)
        
        # Xác định mức độ nghiêm trọng nếu chưa được chỉ định
        if severity == ErrorSeverity.MEDIUM.value:
            severity = self._determine_severity(error, category, error_code)
        
        # Xác định hành động khôi phục
        recovery_action = self._determine_recovery_action(error, category, severity, error_code)
        
        # Tạo bản ghi lỗi
        error_record = {
            'id': f"ERR{int(time.time())}_{component}_{category}",
            'error_type': error.__class__.__name__,
            'message': error_message,
            'traceback': error_traceback,
            'component': component,
            'category': category,
            'severity': severity,
            'error_code': error_code,
            'timestamp': error_time.isoformat(),
            'context': context,
            'recovery_action': recovery_action.value,
            'status': 'new',
            'retry_count': 0,
            'resolved': False,
            'resolution_time': None
        }
        
        # Ghi log lỗi
        log_message = f"Lỗi trong {component}: [{category}] {error_message}"
        
        if severity >= ErrorSeverity.HIGH.value:
            self.logger.error(log_message, exc_info=True)
        elif severity == ErrorSeverity.MEDIUM.value:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Lưu lỗi vào lịch sử
        with self.lock:
            self.error_history.append(error_record)
            
            # Cập nhật số lượng lỗi
            key = f"{component}_{category}"
            self.error_counts[key] = self.error_counts.get(key, 0) + 1
            
            # Thêm vào active errors
            self.active_errors[error_record['id']] = error_record
        
        # Gửi thông báo nếu cần
        self._send_error_notification(error_record)
        
        # Thực hiện hành động khôi phục
        recovery_result = None
        if self.enable_auto_recovery:
            recovery_result = self._perform_recovery(error_record, retry_action)
            error_record['recovery_result'] = recovery_result
        
        # Gọi callback nếu có
        if self.error_callback:
            try:
                self.error_callback(error_record)
            except Exception as callback_error:
                self.logger.error(f"Lỗi trong error_callback: {str(callback_error)}")
        
        # Lưu log lỗi
        self._log_error_to_file(error_record)
        
        return error_record
    
    def resolve_error(self, error_id: str, resolution_notes: Optional[str] = None) -> bool:
        """
        Đánh dấu lỗi đã được giải quyết.
        
        Args:
            error_id: ID lỗi
            resolution_notes: Ghi chú về cách giải quyết
            
        Returns:
            True nếu thành công, False nếu không
        """
        with self.lock:
            if error_id in self.active_errors:
                error_record = self.active_errors[error_id]
                error_record['resolved'] = True
                error_record['resolution_time'] = datetime.now().isoformat()
                error_record['resolution_notes'] = resolution_notes
                error_record['status'] = 'resolved'
                
                # Xóa khỏi active errors
                del self.active_errors[error_id]
                
                self.logger.info(f"Lỗi {error_id} đã được giải quyết")
                return True
            else:
                self.logger.warning(f"Không tìm thấy lỗi {error_id} trong active errors")
                return False
    
    def get_active_errors(
        self,
        component: Optional[str] = None,
        category: Optional[Union[str, ErrorCategory]] = None,
        min_severity: Optional[Union[int, ErrorSeverity]] = None
    ) -> List[Dict[str, Any]]:
        """
        Lấy danh sách lỗi đang hoạt động.
        
        Args:
            component: Lọc theo thành phần
            category: Lọc theo phân loại
            min_severity: Lọc theo mức độ nghiêm trọng tối thiểu
            
        Returns:
            Danh sách lỗi
        """
        # Chuyển đổi enum thành giá trị nếu cần
        if isinstance(category, ErrorCategory):
            category = category.value
            
        if isinstance(min_severity, ErrorSeverity):
            min_severity = min_severity.value
        
        # Lọc lỗi
        filtered_errors = []
        
        with self.lock:
            for error in self.active_errors.values():
                # Kiểm tra điều kiện lọc
                if component and error['component'] != component:
                    continue
                    
                if category and error['category'] != category:
                    continue
                    
                if min_severity and error['severity'] < min_severity:
                    continue
                
                filtered_errors.append(error.copy())
        
        return filtered_errors
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê lỗi.
        
        Returns:
            Dict chứa thống kê lỗi
        """
        with self.lock:
            total_errors = len(self.error_history)
            active_errors = len(self.active_errors)
            resolved_errors = total_errors - active_errors
            
            # Đếm theo phân loại
            category_counts = {}
            severity_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            component_counts = {}
            
            for error in self.error_history:
                # Đếm theo phân loại
                cat = error['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
                
                # Đếm theo mức độ
                sev = error['severity']
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
                
                # Đếm theo thành phần
                comp = error['component']
                component_counts[comp] = component_counts.get(comp, 0) + 1
            
            # Tính thời gian trung bình để giải quyết
            resolution_times = []
            for error in self.error_history:
                if error['resolved'] and error['resolution_time']:
                    start_time = datetime.fromisoformat(error['timestamp'])
                    end_time = datetime.fromisoformat(error['resolution_time'])
                    delta = (end_time - start_time).total_seconds()
                    resolution_times.append(delta)
            
            avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
            
            return {
                'total_errors': total_errors,
                'active_errors': active_errors,
                'resolved_errors': resolved_errors,
                'category_counts': category_counts,
                'severity_counts': severity_counts,
                'component_counts': component_counts,
                'avg_resolution_time': avg_resolution_time
            }
    
    def clear_error_history(self, keep_active: bool = True) -> int:
        """
        Xóa lịch sử lỗi.
        
        Args:
            keep_active: Giữ lại các lỗi đang hoạt động
            
        Returns:
            Số lượng lỗi đã xóa
        """
        with self.lock:
            if keep_active:
                # Lọc ra các lỗi đã giải quyết
                active_ids = set(self.active_errors.keys())
                old_count = len(self.error_history)
                self.error_history = [e for e in self.error_history if e['id'] in active_ids]
                deleted_count = old_count - len(self.error_history)
            else:
                # Xóa tất cả
                deleted_count = len(self.error_history)
                self.error_history = []
                self.active_errors = {}
            
            # Reset số lượng
            self.error_counts = {}
            
            self.logger.info(f"Đã xóa {deleted_count} lỗi khỏi lịch sử")
            return deleted_count
    
    def _classify_error(self, error: Exception, error_message: str) -> str:
        """
        Phân loại lỗi dựa trên thông tin lỗi.
        
        Args:
            error: Exception
            error_message: Thông điệp lỗi
            
        Returns:
            Phân loại lỗi
        """
        error_type = error.__class__.__name__
        error_message_lower = error_message.lower()
        
        # Phân loại dựa trên loại exception
        if error_type in ['ConnectionError', 'ConnectionRefusedError', 'ConnectionResetError', 
                          'TimeoutError', 'socket.timeout']:
            return ErrorCategory.NETWORK.value
            
        elif error_type in ['HTTPError', 'RequestException', 'APIError']:
            return ErrorCategory.API.value
            
        elif error_type in ['DatabaseError', 'OperationalError', 'IntegrityError']:
            return ErrorCategory.DATABASE.value
            
        # Phân loại dựa trên nội dung thông điệp
        if any(kw in error_message_lower for kw in ['api', 'endpoint', 'request']):
            return ErrorCategory.API.value
            
        elif any(kw in error_message_lower for kw in ['network', 'connection', 'timeout', 'connect']):
            return ErrorCategory.NETWORK.value
            
        elif any(kw in error_message_lower for kw in ['exchange', 'binance', 'bybit', 'ftx']):
            return ErrorCategory.EXCHANGE.value
            
        elif any(kw in error_message_lower for kw in ['data', 'dataframe', 'csv', 'json', 'parse']):
            return ErrorCategory.DATA.value
            
        elif any(kw in error_message_lower for kw in ['model', 'predict', 'inference', 'pytorch', 'tensorflow']):
            return ErrorCategory.MODEL.value
            
        elif any(kw in error_message_lower for kw in ['database', 'sql', 'query', 'db']):
            return ErrorCategory.DATABASE.value
            
        elif any(kw in error_message_lower for kw in ['auth', 'token', 'credential', 'permission']):
            return ErrorCategory.AUTHENTICATION.value
            
        elif any(kw in error_message_lower for kw in ['rate limit', 'too many requests', 'throttle']):
            return ErrorCategory.RATE_LIMIT.value
            
        # Mặc định
        return ErrorCategory.UNKNOWN.value
    
    def _determine_severity(self, error: Exception, category: str, error_code: int) -> int:
        """
        Xác định mức độ nghiêm trọng của lỗi.
        
        Args:
            error: Exception
            category: Phân loại lỗi
            error_code: Mã lỗi
            
        Returns:
            Mức độ nghiêm trọng
        """
        error_type = error.__class__.__name__
        
        # Lỗi nghiêm trọng
        if category in [ErrorCategory.AUTHENTICATION.value, ErrorCategory.DATABASE.value]:
            return ErrorSeverity.HIGH.value
            
        # Lỗi trung bình
        if category in [ErrorCategory.NETWORK.value, ErrorCategory.API.value, 
                       ErrorCategory.EXCHANGE.value, ErrorCategory.RATE_LIMIT.value]:
            return ErrorSeverity.MEDIUM.value
            
        # Lỗi nhẹ
        if category in [ErrorCategory.DATA.value]:
            return ErrorSeverity.LOW.value
            
        # Phân loại dựa trên loại exception
        if error_type in ['SystemExit', 'KeyboardInterrupt', 'SystemError']:
            return ErrorSeverity.CRITICAL.value
            
        elif error_type in ['MemoryError', 'OverflowError', 'RecursionError']:
            return ErrorSeverity.HIGH.value
            
        # Mặc định
        return ErrorSeverity.MEDIUM.value
    
    def _determine_recovery_action(
        self,
        error: Exception,
        category: str,
        severity: int,
        error_code: int
    ) -> RecoveryAction:
        """
        Xác định hành động khôi phục.
        
        Args:
            error: Exception
            category: Phân loại lỗi
            severity: Mức độ nghiêm trọng
            error_code: Mã lỗi
            
        Returns:
            Hành động khôi phục
        """
        # Dựa trên mức độ nghiêm trọng
        if severity >= ErrorSeverity.CRITICAL.value:
            return RecoveryAction.EMERGENCY_STOP
            
        elif severity >= ErrorSeverity.HIGH.value:
            if category in [ErrorCategory.AUTHENTICATION.value, ErrorCategory.DATABASE.value]:
                return RecoveryAction.PAUSE_TRADING
            else:
                return RecoveryAction.RESTART_COMPONENT
            
        elif severity >= ErrorSeverity.MEDIUM.value:
            if category in [ErrorCategory.NETWORK.value, ErrorCategory.API.value, 
                          ErrorCategory.RATE_LIMIT.value]:
                return RecoveryAction.RETRY
            else:
                return RecoveryAction.RESTART_COMPONENT
            
        # Mặc định
        return RecoveryAction.RETRY if category != ErrorCategory.UNKNOWN.value else RecoveryAction.NONE
    
    def _send_error_notification(self, error_record: Dict[str, Any]) -> None:
        """
        Gửi thông báo về lỗi.
        
        Args:
            error_record: Bản ghi lỗi
        """
        if not self.notification_manager:
            return
            
        # Xác định mức độ ưu tiên
        priority = NotificationPriority.NORMAL
        if error_record['severity'] >= ErrorSeverity.CRITICAL.value:
            priority = NotificationPriority.CRITICAL
        elif error_record['severity'] >= ErrorSeverity.HIGH.value:
            priority = NotificationPriority.HIGH
            
        # Tạo thông điệp
        subject = f"Lỗi {error_record['category'].upper()} trong {error_record['component']}"
        message = f"ID Lỗi: {error_record['id']}\n"
        message += f"Loại: {error_record['error_type']}\n"
        message += f"Thông điệp: {error_record['message']}\n"
        message += f"Thành phần: {error_record['component']}\n"
        message += f"Phân loại: {error_record['category']}\n"
        message += f"Mức độ: {error_record['severity']}\n"
        message += f"Thời gian: {error_record['timestamp']}\n"
        message += f"Hành động khôi phục: {error_record['recovery_action']}\n"
        
        if error_record['context']:
            message += f"\nThông tin ngữ cảnh:\n{error_record['context']}\n"
            
        # Thêm stacktrace cho lỗi nghiêm trọng
        if error_record['severity'] >= ErrorSeverity.HIGH.value:
            message += f"\nChi tiết lỗi:\n{error_record['traceback']}"
            
        # Gửi thông báo
        try:
            self.notification_manager.send_error_notification(
                error_message=error_record['message'],
                error_source=error_record['component'],
                error_trace=error_record['traceback'] if error_record['severity'] >= ErrorSeverity.HIGH.value else None,
                critical=error_record['severity'] >= ErrorSeverity.HIGH.value
            )
        except Exception as e:
            self.logger.error(f"Không thể gửi thông báo lỗi: {str(e)}")
    
    def _perform_recovery(
        self,
        error_record: Dict[str, Any],
        retry_action: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Thực hiện hành động khôi phục.
        
        Args:
            error_record: Bản ghi lỗi
            retry_action: Hàm thử lại
            
        Returns:
            Kết quả khôi phục
        """
        recovery_action = error_record['recovery_action']
        
        # Kết quả mặc định
        result = {
            'action': recovery_action,
            'success': False,
            'message': 'Không thực hiện hành động',
            'timestamp': datetime.now().isoformat()
        }
        
        # Nếu không có hành động
        if recovery_action == RecoveryAction.NONE.value:
            result['message'] = 'Không cần hành động khôi phục'
            result['success'] = True
            return result
            
        # Nếu là thử lại
        elif recovery_action == RecoveryAction.RETRY.value:
            if retry_action and callable(retry_action):
                max_retries = self.max_retries
                retry_delay = self.retry_delay
                
                for retry_count in range(1, max_retries + 1):
                    try:
                        # Cập nhật trạng thái
                        error_record['retry_count'] = retry_count
                        error_record['status'] = f'retrying_{retry_count}'
                        
                        self.logger.info(f"Đang thử lại [{retry_count}/{max_retries}] cho lỗi {error_record['id']}")
                        
                        # Thực hiện thử lại
                        retry_result = retry_action()
                        
                        # Nếu thành công
                        error_record['status'] = 'resolved_by_retry'
                        error_record['resolved'] = True
                        error_record['resolution_time'] = datetime.now().isoformat()
                        
                        # Xóa khỏi active errors
                        with self.lock:
                            if error_record['id'] in self.active_errors:
                                del self.active_errors[error_record['id']]
                        
                        result['success'] = True
                        result['message'] = f'Đã thử lại thành công sau {retry_count} lần'
                        
                        self.logger.info(f"Đã khôi phục thành công lỗi {error_record['id']} sau {retry_count} lần thử lại")
                        
                        return result
                        
                    except Exception as e:
                        self.logger.warning(f"Thử lại lần {retry_count} không thành công: {str(e)}")
                        # Đợi trước khi thử lại
                        time.sleep(retry_delay)
                
                # Nếu tất cả các lần thử đều thất bại
                result['message'] = f'Đã thử lại {max_retries} lần không thành công'
                return result
            else:
                result['message'] = 'Không có hàm thử lại được cung cấp'
                return result
                
        # Các hành động khác cần được triển khai thông qua recovery_system
        else:
            result['message'] = f'Hành động {recovery_action} cần được thực hiện thông qua recovery_system'
            return result
    
    def _log_error_to_file(self, error_record: Dict[str, Any]) -> None:
        """
        Lưu lỗi vào file.
        
        Args:
            error_record: Bản ghi lỗi
        """
        try:
            # Tạo tên file dựa trên ngày
            date_str = datetime.now().strftime('%Y%m%d')
            file_path = os.path.join(self.error_log_dir, f"errors_{date_str}.log")
            
            # Tạo nội dung log
            log_content = f"[{error_record['timestamp']}] ERROR {error_record['id']}: "
            log_content += f"{error_record['component']} - {error_record['category']} - "
            log_content += f"{error_record['message']}\n"
            
            if error_record['severity'] >= ErrorSeverity.HIGH.value:
                log_content += f"Traceback:\n{error_record['traceback']}\n"
                
            log_content += f"Severity: {error_record['severity']}, "
            log_content += f"Recovery: {error_record['recovery_action']}\n"
            log_content += "-" * 80 + "\n"
            
            # Ghi vào file
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(log_content)
                
        except Exception as e:
            self.logger.error(f"Không thể lưu lỗi vào file: {str(e)}")
    
    def retry_with_backoff(
        self,
        func: Callable,
        max_retries: Optional[int] = None,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        exceptions: Tuple[Exception] = (Exception,),
        retry_callback: Optional[Callable] = None
    ):
        """
        Thực hiện hàm với cơ chế thử lại có trễ tăng dần.
        
        Args:
            func: Hàm cần thực hiện
            max_retries: Số lần thử lại tối đa (None để dùng giá trị mặc định)
            initial_delay: Thời gian trễ ban đầu (giây)
            backoff_factor: Hệ số tăng thời gian trễ
            max_delay: Thời gian trễ tối đa (giây)
            exceptions: Tuple các loại exception được thử lại
            retry_callback: Hàm callback khi thử lại
            
        Returns:
            Kết quả của hàm
        """
        if max_retries is None:
            max_retries = self.max_retries
            
        retries = 0
        delay = initial_delay
        
        while True:
            try:
                return func()
            except exceptions as e:
                retries += 1
                
                if retries > max_retries:
                    # Hết số lần thử lại
                    self.logger.error(f"Đã đạt đến số lần thử lại tối đa ({max_retries})")
                    raise
                
                # Tính thời gian trễ tiếp theo
                delay = min(delay * backoff_factor, max_delay)
                
                # Ghi log
                self.logger.warning(f"Thử lại lần {retries}/{max_retries} sau {delay:.1f}s. Lỗi: {str(e)}")
                
                # Gọi callback nếu có
                if retry_callback:
                    try:
                        retry_callback(e, retries, delay)
                    except Exception as callback_error:
                        self.logger.error(f"Lỗi trong retry_callback: {str(callback_error)}")
                
                # Đợi trước khi thử lại
                time.sleep(delay)