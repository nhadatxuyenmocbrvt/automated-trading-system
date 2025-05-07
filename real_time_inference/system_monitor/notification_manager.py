"""
Quản lý thông báo hệ thống.
File này cung cấp lớp NotificationManager để quản lý và gửi thông báo
qua nhiều kênh khác nhau (email, Telegram, v.v.) với khả năng phân loại
mức độ ưu tiên và định tuyến thông báo.
"""

import os
import time
import json
import logging
import threading
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from enum import Enum
from pathlib import Path
from datetime import datetime

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from config.security_config import get_security_config

# Import các notifier
try:
    from real_time_inference.notifiers.email_notifier import EmailNotifier
    EMAIL_NOTIFIER_AVAILABLE = True
except ImportError:
    EMAIL_NOTIFIER_AVAILABLE = False

try:
    from real_time_inference.notifiers.telegram_notifier import TelegramNotifier
    TELEGRAM_NOTIFIER_AVAILABLE = True
except ImportError:
    TELEGRAM_NOTIFIER_AVAILABLE = False

class NotificationPriority(Enum):
    """Mức độ ưu tiên của thông báo."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class NotificationType(Enum):
    """Loại thông báo."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    TRADE = "trade"
    SYSTEM = "system"
    ALERT = "alert"

class NotificationChannel(Enum):
    """Kênh thông báo."""
    EMAIL = "email"
    TELEGRAM = "telegram"
    LOG = "log"
    ALL = "all"

class NotificationManager:
    """
    Lớp quản lý thông báo.
    Cung cấp các phương thức để gửi thông báo qua nhiều kênh khác nhau,
    quản lý ưu tiên, và xử lý sự cố khi gửi thông báo.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_email: bool = False,
        enable_telegram: bool = False,
        rate_limit: int = 60,  # Số thông báo tối đa trong 1 phút
        notification_log_dir: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo NotificationManager.
        
        Args:
            config: Cấu hình tùy chỉnh
            enable_email: Kích hoạt thông báo qua email
            enable_telegram: Kích hoạt thông báo qua Telegram
            rate_limit: Giới hạn số thông báo trong 1 phút
            notification_log_dir: Thư mục lưu log thông báo
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("notification_manager")
        
        # Lấy cấu hình hệ thống nếu không được cung cấp
        if config is None:
            self.config = get_system_config().get_all()
            # Lấy cấu hình thông báo cụ thể nếu có
            notification_config = self.config.get("notifications", {})
            enable_email = notification_config.get("enable_email", enable_email)
            enable_telegram = notification_config.get("enable_telegram", enable_telegram)
            rate_limit = notification_config.get("rate_limit", rate_limit)
        else:
            self.config = config
        
        # Thiết lập các thông số cơ bản
        self.rate_limit = rate_limit
        self.notification_count = 0
        self.last_reset_time = time.time()
        
        # Thư mục lưu log thông báo
        if notification_log_dir is None:
            base_log_dir = Path(self.config.get("log_dir", "logs"))
            self.notification_log_dir = base_log_dir / "notifications"
        else:
            self.notification_log_dir = Path(notification_log_dir)
        
        # Đảm bảo thư mục tồn tại
        self.notification_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo các notifier
        self.notifiers = {}
        
        # Khởi tạo EmailNotifier nếu được kích hoạt và có sẵn
        if enable_email and EMAIL_NOTIFIER_AVAILABLE:
            try:
                self.notifiers[NotificationChannel.EMAIL.value] = EmailNotifier()
                self.logger.info("Đã khởi tạo EmailNotifier")
            except Exception as e:
                self.logger.error(f"Không thể khởi tạo EmailNotifier: {str(e)}")
        
        # Khởi tạo TelegramNotifier nếu được kích hoạt và có sẵn
        if enable_telegram and TELEGRAM_NOTIFIER_AVAILABLE:
            try:
                self.notifiers[NotificationChannel.TELEGRAM.value] = TelegramNotifier()
                self.logger.info("Đã khởi tạo TelegramNotifier")
            except Exception as e:
                self.logger.error(f"Không thể khởi tạo TelegramNotifier: {str(e)}")
        
        # Thiết lập trạng thái
        self.active = True
        self.notification_history = []
        self.notification_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Tạo luồng xử lý hàng đợi thông báo
        self.notification_thread = threading.Thread(
            target=self._process_notification_queue,
            daemon=True,
            name="NotificationThread"
        )
        self.notification_thread.start()
        
        self.logger.info(f"Đã khởi tạo NotificationManager với {len(self.notifiers)} kênh thông báo")
    
    def send_notification(
        self,
        message: str,
        subject: Optional[str] = None,
        notification_type: Union[str, NotificationType] = NotificationType.INFO,
        priority: Union[int, NotificationPriority] = NotificationPriority.NORMAL,
        channels: Optional[List[Union[str, NotificationChannel]]] = None,
        attachments: Optional[List[Union[str, Path]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        immediate: bool = False
    ) -> bool:
        """
        Gửi thông báo qua các kênh được chỉ định.
        
        Args:
            message: Nội dung thông báo
            subject: Tiêu đề thông báo (tự động tạo nếu không cung cấp)
            notification_type: Loại thông báo
            priority: Mức độ ưu tiên
            channels: Danh sách kênh gửi (None để sử dụng tất cả)
            attachments: Danh sách tệp đính kèm
            metadata: Dữ liệu bổ sung
            immediate: Gửi ngay lập tức không qua hàng đợi
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        # Kiểm tra xem hệ thống có đang hoạt động không
        if not self.active:
            self.logger.warning(f"Không thể gửi thông báo: Hệ thống không hoạt động")
            return False
        
        # Kiểm tra rate limit
        current_time = time.time()
        if current_time - self.last_reset_time > 60:
            # Reset counter mỗi phút
            self.notification_count = 0
            self.last_reset_time = current_time
        
        if self.notification_count >= self.rate_limit:
            self.logger.warning(f"Đã đạt giới hạn thông báo ({self.rate_limit}/phút)")
            return False
        
        # Chuyển đổi các enum thành string nếu cần
        if isinstance(notification_type, NotificationType):
            notification_type = notification_type.value
            
        if isinstance(priority, NotificationPriority):
            priority = priority.value
        
        # Tạo tiêu đề tự động nếu không cung cấp
        if subject is None:
            subject = f"{notification_type.upper()} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Chuẩn hóa danh sách kênh
        if channels is None:
            channels = list(self.notifiers.keys())
        else:
            normalized_channels = []
            for channel in channels:
                if isinstance(channel, NotificationChannel):
                    channel_value = channel.value
                else:
                    channel_value = channel
                
                if channel_value == NotificationChannel.ALL.value:
                    normalized_channels = list(self.notifiers.keys())
                    break
                elif channel_value in self.notifiers:
                    normalized_channels.append(channel_value)
            
            channels = normalized_channels
        
        # Nếu không có kênh nào khả dụng, chỉ ghi log
        if not channels:
            self.logger.warning(f"Không có kênh thông báo nào khả dụng")
            self._log_notification(message, subject, notification_type, priority, metadata)
            return False
        
        # Tạo đối tượng thông báo
        notification = {
            'message': message,
            'subject': subject,
            'type': notification_type,
            'priority': priority,
            'channels': channels,
            'attachments': [str(attachment) for attachment in attachments] if attachments else None,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        # Ghi log thông báo
        log_level = logging.INFO
        if priority >= NotificationPriority.HIGH.value:
            log_level = logging.WARNING
        if priority >= NotificationPriority.CRITICAL.value:
            log_level = logging.ERROR
            
        self.logger.log(log_level, f"Thông báo [{notification_type}]: {subject} - {message}")
        
        # Xử lý thông báo
        if immediate:
            success = self._send_notification_to_channels(notification)
            # Tăng counter
            self.notification_count += 1
            return success
        else:
            # Thêm vào buffer
            with self.buffer_lock:
                self.notification_buffer.append(notification)
            # Tăng counter
            self.notification_count += 1
            return True
    
    def _send_notification_to_channels(self, notification: Dict[str, Any]) -> bool:
        """
        Gửi thông báo qua các kênh được chỉ định.
        
        Args:
            notification: Thông tin thông báo
            
        Returns:
            True nếu ít nhất một kênh gửi thành công, False nếu tất cả đều thất bại
        """
        message = notification['message']
        subject = notification['subject']
        notification_type = notification['type']
        channels = notification['channels']
        attachments = notification['attachments']
        metadata = notification['metadata']
        
        # Ghi log thông báo
        self._log_notification(message, subject, notification_type, notification['priority'], metadata)
        
        # Theo dõi thành công
        success = False
        results = {}
        
        # Gửi qua từng kênh
        for channel in channels:
            if channel in self.notifiers:
                try:
                    notifier = self.notifiers[channel]
                    # Gọi phương thức gửi thông báo tương ứng
                    if channel == NotificationChannel.EMAIL.value:
                        result = notifier.send_email(
                            subject=subject,
                            message=message,
                            attachments=attachments,
                            notification_type=notification_type
                        )
                    elif channel == NotificationChannel.TELEGRAM.value:
                        result = notifier.send_message(
                            message=message,
                            notification_type=notification_type
                        )
                    else:
                        # Gọi hàm send chung nếu không có xử lý đặc biệt
                        result = notifier.send(
                            subject=subject,
                            message=message,
                            notification_type=notification_type,
                            attachments=attachments
                        )
                    
                    # Cập nhật kết quả
                    results[channel] = result
                    if result:
                        success = True
                        
                except Exception as e:
                    self.logger.error(f"Lỗi khi gửi thông báo qua {channel}: {str(e)}")
                    results[channel] = False
        
        # Cập nhật lịch sử
        notification['status'] = 'success' if success else 'failed'
        notification['results'] = results
        notification['sent_time'] = datetime.now().isoformat()
        
        with self.buffer_lock:
            self.notification_history.append(notification)
            
            # Giữ lịch sử thông báo ở mức hợp lý
            if len(self.notification_history) > 1000:
                # Xóa 10% các thông báo cũ nhất
                self.notification_history = self.notification_history[-900:]
        
        return success
    
    def _log_notification(
        self,
        message: str,
        subject: str,
        notification_type: str,
        priority: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Ghi log thông báo vào file.
        
        Args:
            message: Nội dung thông báo
            subject: Tiêu đề thông báo
            notification_type: Loại thông báo
            priority: Mức độ ưu tiên
            metadata: Dữ liệu bổ sung
        """
        try:
            # Tạo bản ghi log
            log_entry = {
                'message': message,
                'subject': subject,
                'type': notification_type,
                'priority': priority,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat()
            }
            
            # Xác định tên file log dựa trên ngày
            log_file = self.notification_log_dir / f"notifications_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            # Ghi vào file
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            self.logger.error(f"Lỗi khi ghi log thông báo: {str(e)}")
    
    def _process_notification_queue(self) -> None:
        """
        Xử lý hàng đợi thông báo.
        Chạy trong một luồng riêng để xử lý các thông báo theo hàng đợi.
        """
        while self.active:
            # Kiểm tra và xử lý thông báo trong buffer
            with self.buffer_lock:
                if self.notification_buffer:
                    # Lấy thông báo đầu tiên
                    notification = self.notification_buffer.pop(0)
                else:
                    notification = None
            
            # Xử lý thông báo nếu có
            if notification:
                self._send_notification_to_channels(notification)
            
            # Nghỉ một chút để tránh tốn CPU
            time.sleep(0.1)
    
    def send_error_notification(
        self,
        error_message: str,
        error_source: str,
        error_trace: Optional[str] = None,
        critical: bool = False,
        immediate: bool = True
    ) -> bool:
        """
        Gửi thông báo lỗi.
        
        Args:
            error_message: Thông điệp lỗi
            error_source: Nguồn gốc lỗi
            error_trace: Stack trace lỗi
            critical: Lỗi có nghiêm trọng không
            immediate: Gửi ngay lập tức
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        subject = f"Lỗi hệ thống từ {error_source}"
        
        # Tạo nội dung thông báo
        message = f"Lỗi: {error_message}\n"
        message += f"Nguồn: {error_source}\n"
        
        if error_trace:
            message += f"\nChi tiết lỗi:\n{error_trace}"
        
        # Xác định mức độ ưu tiên
        priority = NotificationPriority.HIGH
        if critical:
            priority = NotificationPriority.CRITICAL
        
        # Gửi thông báo
        return self.send_notification(
            message=message,
            subject=subject,
            notification_type=NotificationType.ERROR,
            priority=priority,
            immediate=immediate,
            metadata={
                'error_source': error_source,
                'has_trace': error_trace is not None,
                'critical': critical
            }
        )
    
    def send_trade_notification(
        self,
        trade_details: Dict[str, Any],
        success: bool = True,
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> bool:
        """
        Gửi thông báo giao dịch.
        
        Args:
            trade_details: Chi tiết giao dịch
            success: Giao dịch thành công hay không
            priority: Mức độ ưu tiên
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        # Trích xuất thông tin
        symbol = trade_details.get('symbol', 'Unknown')
        side = trade_details.get('side', 'Unknown')
        quantity = trade_details.get('quantity', 0)
        price = trade_details.get('price', 0)
        order_type = trade_details.get('order_type', 'Unknown')
        
        # Tạo tiêu đề
        action = "Thành công" if success else "Thất bại"
        subject = f"Giao dịch {action}: {side.upper()} {symbol}"
        
        # Tạo nội dung
        message = f"Chi tiết giao dịch:\n"
        message += f"Symbol: {symbol}\n"
        message += f"Loại lệnh: {order_type.upper()} {side.upper()}\n"
        message += f"Số lượng: {quantity}\n"
        message += f"Giá: {price}\n"
        
        # Thêm thông tin P&L nếu có
        if 'pnl' in trade_details:
            message += f"P&L: {trade_details['pnl']}\n"
        
        # Thêm lý do nếu thất bại
        if not success and 'reason' in trade_details:
            message += f"Lý do: {trade_details['reason']}\n"
        
        # Loại thông báo
        notification_type = NotificationType.SUCCESS if success else NotificationType.ERROR
        
        # Gửi thông báo
        return self.send_notification(
            message=message,
            subject=subject,
            notification_type=notification_type,
            priority=priority,
            metadata=trade_details
        )
    
    def send_system_notification(
        self,
        component: str,
        status: str,
        details: Optional[str] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> bool:
        """
        Gửi thông báo trạng thái hệ thống.
        
        Args:
            component: Tên thành phần
            status: Trạng thái (started, stopped, error, etc.)
            details: Chi tiết bổ sung
            priority: Mức độ ưu tiên
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        # Tạo tiêu đề
        subject = f"Hệ thống: {component} - {status.upper()}"
        
        # Tạo nội dung
        message = f"Thành phần: {component}\n"
        message += f"Trạng thái: {status.upper()}\n"
        
        if details:
            message += f"Chi tiết: {details}\n"
        
        # Thêm timestamp
        message += f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Xác định loại thông báo
        notification_type = NotificationType.INFO
        if status.lower() in ['error', 'failed', 'crash', 'critical']:
            notification_type = NotificationType.ERROR
        elif status.lower() in ['warning', 'unstable']:
            notification_type = NotificationType.WARNING
        elif status.lower() in ['success', 'started', 'completed']:
            notification_type = NotificationType.SUCCESS
        
        # Gửi thông báo
        return self.send_notification(
            message=message,
            subject=subject,
            notification_type=notification_type,
            priority=priority,
            metadata={
                'component': component,
                'status': status,
                'has_details': details is not None
            }
        )
    
    def send_price_alert(
        self,
        symbol: str,
        current_price: float,
        alert_type: str,
        threshold: float,
        priority: NotificationPriority = NotificationPriority.HIGH
    ) -> bool:
        """
        Gửi thông báo cảnh báo giá.
        
        Args:
            symbol: Symbol giao dịch
            current_price: Giá hiện tại
            alert_type: Loại cảnh báo (above, below, volatility)
            threshold: Ngưỡng giá
            priority: Mức độ ưu tiên
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        # Tạo tiêu đề
        subject = f"Cảnh báo giá {symbol}: {alert_type.upper()}"
        
        # Tạo nội dung
        message = f"Cảnh báo giá cho {symbol}\n"
        message += f"Giá hiện tại: {current_price}\n"
        
        if alert_type.lower() == 'above':
            message += f"Giá đã vượt ngưỡng: {threshold}\n"
        elif alert_type.lower() == 'below':
            message += f"Giá đã giảm xuống dưới ngưỡng: {threshold}\n"
        elif alert_type.lower() == 'volatility':
            message += f"Biến động giá đã vượt ngưỡng: {threshold}%\n"
        
        # Thêm timestamp
        message += f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Gửi thông báo
        return self.send_notification(
            message=message,
            subject=subject,
            notification_type=NotificationType.ALERT,
            priority=priority,
            metadata={
                'symbol': symbol,
                'current_price': current_price,
                'alert_type': alert_type,
                'threshold': threshold
            }
        )
    
    def add_notifier(self, channel: str, notifier: Any) -> None:
        """
        Thêm notifier mới.
        
        Args:
            channel: Tên kênh
            notifier: Đối tượng notifier
        """
        self.notifiers[channel] = notifier
        self.logger.info(f"Đã thêm notifier mới: {channel}")
    
    def remove_notifier(self, channel: str) -> bool:
        """
        Xóa notifier.
        
        Args:
            channel: Tên kênh
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        if channel in self.notifiers:
            del self.notifiers[channel]
            self.logger.info(f"Đã xóa notifier: {channel}")
            return True
        else:
            self.logger.warning(f"Không tìm thấy notifier: {channel}")
            return False
    
    def get_notification_history(
        self,
        limit: int = 100,
        notification_type: Optional[str] = None,
        min_priority: Optional[int] = None,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử thông báo.
        
        Args:
            limit: Số lượng thông báo tối đa
            notification_type: Lọc theo loại thông báo
            min_priority: Lọc theo mức độ ưu tiên tối thiểu
            start_time: Thời gian bắt đầu
            end_time: Thời gian kết thúc
            
        Returns:
            Danh sách thông báo
        """
        with self.buffer_lock:
            # Sao chép lịch sử để tránh race condition
            history = self.notification_history.copy()
        
        # Chuyển đổi timestamp sang đối tượng datetime nếu cần
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time)
        
        # Lọc theo các tiêu chí
        filtered_history = []
        for notification in reversed(history):  # Đảo ngược để lấy mới nhất trước
            # Kiểm tra loại thông báo
            if notification_type and notification['type'] != notification_type:
                continue
            
            # Kiểm tra mức độ ưu tiên
            if min_priority and notification['priority'] < min_priority:
                continue
            
            # Kiểm tra thời gian bắt đầu
            if start_time:
                notification_time = datetime.fromisoformat(notification['timestamp'])
                if notification_time < start_time:
                    continue
            
            # Kiểm tra thời gian kết thúc
            if end_time:
                notification_time = datetime.fromisoformat(notification['timestamp'])
                if notification_time > end_time:
                    continue
            
            # Thêm vào danh sách lọc
            filtered_history.append(notification)
            
            # Kiểm tra giới hạn
            if len(filtered_history) >= limit:
                break
        
        return filtered_history
    
    def clear_notification_history(self) -> None:
        """Xóa lịch sử thông báo."""
        with self.buffer_lock:
            self.notification_history = []
            self.logger.info("Đã xóa lịch sử thông báo")
    
    def set_rate_limit(self, limit: int) -> None:
        """
        Đặt giới hạn tốc độ gửi thông báo.
        
        Args:
            limit: Số thông báo tối đa trong 1 phút
        """
        self.rate_limit = limit
        self.logger.info(f"Đã đặt giới hạn thông báo: {limit}/phút")
    
    def shutdown(self) -> None:
        """Tắt hệ thống thông báo."""
        self.logger.info("Đang tắt hệ thống thông báo...")
        
        # Đặt trạng thái không hoạt động
        self.active = False
        
        # Đợi luồng xử lý kết thúc
        if self.notification_thread.is_alive():
            self.notification_thread.join(timeout=5.0)
        
        # Xử lý các thông báo còn lại trong buffer
        with self.buffer_lock:
            remaining = len(self.notification_buffer)
            if remaining > 0:
                self.logger.warning(f"Còn {remaining} thông báo chưa gửi")
        
        # Đóng các notifier nếu có phương thức close
        for channel, notifier in self.notifiers.items():
            if hasattr(notifier, 'close'):
                try:
                    notifier.close()
                except Exception as e:
                    self.logger.error(f"Lỗi khi đóng notifier {channel}: {str(e)}")
        
        self.logger.info("Đã tắt hệ thống thông báo")


# Tạo instance mặc định để sử dụng trong ứng dụng
def get_notification_manager(
    config: Optional[Dict[str, Any]] = None,
    enable_email: bool = False,
    enable_telegram: bool = False
) -> NotificationManager:
    """
    Tạo hoặc lấy instance NotificationManager.
    
    Args:
        config: Cấu hình tùy chỉnh
        enable_email: Kích hoạt thông báo qua email
        enable_telegram: Kích hoạt thông báo qua Telegram
        
    Returns:
        Instance NotificationManager
    """
    # Lưu ý: Đây sẽ tạo mới một instance mỗi lần gọi hàm này
    # Trong ứng dụng thực tế, bạn có thể muốn sử dụng Singleton pattern
    return NotificationManager(
        config=config,
        enable_email=enable_email,
        enable_telegram=enable_telegram
    )