"""
Hệ thống cảnh báo.
File này cung cấp các lớp và hàm để phát hiện, xử lý và gửi cảnh báo khi
hệ thống giao dịch gặp vấn đề hoặc đạt đến ngưỡng giám sát quan trọng.
"""

import os
import time
import logging
import threading
import json
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Set
from pathlib import Path
from enum import Enum, auto
import asyncio

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from logs.logger import SystemLogger

class AlertLevel(Enum):
    """Cấp độ cảnh báo."""
    INFO = auto()      # Thông tin thông thường
    NOTICE = auto()    # Thông báo cần chú ý
    WARNING = auto()   # Cảnh báo, cần xem xét
    ALERT = auto()     # Cảnh báo nghiêm trọng, cần xử lý sớm
    CRITICAL = auto()  # Cực kỳ nghiêm trọng, cần xử lý ngay lập tức

class AlertCategory(Enum):
    """Phân loại cảnh báo."""
    SYSTEM = auto()      # Liên quan đến hệ thống (CPU, RAM, disk)
    NETWORK = auto()     # Liên quan đến mạng và kết nối
    SECURITY = auto()    # Liên quan đến bảo mật
    TRADING = auto()     # Liên quan đến giao dịch
    FINANCIAL = auto()   # Liên quan đến tài chính
    PERFORMANCE = auto() # Liên quan đến hiệu suất
    API = auto()         # Liên quan đến API
    DATABASE = auto()    # Liên quan đến cơ sở dữ liệu
    CUSTOM = auto()      # Loại tùy chỉnh khác

class Alert:
    """
    Lớp đại diện cho một cảnh báo.
    Chứa thông tin về một cảnh báo cụ thể, bao gồm cấp độ,
    thông điệp, thời gian, và các dữ liệu liên quan.
    """
    
    def __init__(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        category: AlertCategory = AlertCategory.SYSTEM,
        source: str = "system",
        data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        alert_id: Optional[str] = None
    ):
        """
        Khởi tạo cảnh báo mới.
        
        Args:
            message: Thông điệp cảnh báo
            level: Cấp độ cảnh báo
            category: Loại cảnh báo
            source: Nguồn phát sinh cảnh báo
            data: Dữ liệu bổ sung
            timestamp: Thời gian phát sinh (None để sử dụng thời gian hiện tại)
            alert_id: ID cảnh báo (None để tự động tạo)
        """
        self.message = message
        self.level = level
        self.category = category
        self.source = source
        self.data = data or {}
        self.timestamp = timestamp or datetime.now()
        self.alert_id = alert_id or self._generate_id()
        
        # Trạng thái xử lý
        self.acknowledged = False
        self.resolved = False
        self.acknowledged_time = None
        self.resolved_time = None
        self.acknowledged_by = None
        self.resolved_by = None
        self.notes = []
    
    def _generate_id(self) -> str:
        """
        Tạo ID duy nhất cho cảnh báo.
        
        Returns:
            ID cảnh báo
        """
        import uuid
        timestamp_str = self.timestamp.strftime("%Y%m%d%H%M%S")
        unique_part = str(uuid.uuid4())[:8]
        return f"ALERT-{self.level.name}-{timestamp_str}-{unique_part}"
    
    def acknowledge(self, by: Optional[str] = None) -> None:
        """
        Xác nhận đã biết về cảnh báo.
        
        Args:
            by: Người xác nhận
        """
        self.acknowledged = True
        self.acknowledged_time = datetime.now()
        self.acknowledged_by = by
    
    def resolve(self, by: Optional[str] = None) -> None:
        """
        Đánh dấu cảnh báo đã được giải quyết.
        
        Args:
            by: Người giải quyết
        """
        self.resolved = True
        self.resolved_time = datetime.now()
        self.resolved_by = by
    
    def add_note(self, note: str, by: Optional[str] = None) -> None:
        """
        Thêm ghi chú cho cảnh báo.
        
        Args:
            note: Nội dung ghi chú
            by: Người ghi chú
        """
        self.notes.append({
            "note": note,
            "timestamp": datetime.now(),
            "by": by
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thành dictionary.
        
        Returns:
            Dict chứa thông tin cảnh báo
        """
        return {
            "alert_id": self.alert_id,
            "message": self.message,
            "level": self.level.name,
            "category": self.category.name,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "acknowledged_time": self.acknowledged_time.isoformat() if self.acknowledged_time else None,
            "resolved_time": self.resolved_time.isoformat() if self.resolved_time else None,
            "acknowledged_by": self.acknowledged_by,
            "resolved_by": self.resolved_by,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """
        Tạo đối tượng Alert từ dictionary.
        
        Args:
            data: Dictionary chứa thông tin cảnh báo
            
        Returns:
            Đối tượng Alert
        """
        alert = cls(
            message=data["message"],
            level=AlertLevel[data["level"]],
            category=AlertCategory[data["category"]],
            source=data["source"],
            data=data.get("data", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            alert_id=data["alert_id"]
        )
        
        # Khôi phục trạng thái
        alert.acknowledged = data.get("acknowledged", False)
        alert.resolved = data.get("resolved", False)
        
        if data.get("acknowledged_time"):
            alert.acknowledged_time = datetime.fromisoformat(data["acknowledged_time"])
        
        if data.get("resolved_time"):
            alert.resolved_time = datetime.fromisoformat(data["resolved_time"])
        
        alert.acknowledged_by = data.get("acknowledged_by")
        alert.resolved_by = data.get("resolved_by")
        alert.notes = data.get("notes", [])
        
        return alert

class AlertHandler:
    """
    Lớp xử lý cảnh báo cơ bản.
    Định nghĩa giao diện cho các handler khác kế thừa.
    """
    
    def __init__(self, name: str, min_level: AlertLevel = AlertLevel.INFO):
        """
        Khởi tạo handler.
        
        Args:
            name: Tên handler
            min_level: Cấp độ cảnh báo tối thiểu để xử lý
        """
        self.name = name
        self.min_level = min_level
        self.logger = SystemLogger(f"alert_handler_{name}")
    
    def can_handle(self, alert: Alert) -> bool:
        """
        Kiểm tra xem handler có thể xử lý cảnh báo hay không.
        
        Args:
            alert: Cảnh báo cần kiểm tra
            
        Returns:
            True nếu có thể xử lý, False nếu không
        """
        return alert.level.value >= self.min_level.value
    
    async def handle(self, alert: Alert) -> bool:
        """
        Xử lý cảnh báo.
        
        Args:
            alert: Cảnh báo cần xử lý
            
        Returns:
            True nếu xử lý thành công, False nếu không
        """
        # Phương thức ảo, các lớp con sẽ ghi đè
        return False

class ConsoleAlertHandler(AlertHandler):
    """
    Handler hiển thị cảnh báo trên console.
    """
    
    async def handle(self, alert: Alert) -> bool:
        """
        Xử lý cảnh báo bằng cách ghi log ra console.
        
        Args:
            alert: Cảnh báo cần xử lý
            
        Returns:
            True nếu xử lý thành công, False nếu không
        """
        # Ánh xạ cấp độ cảnh báo sang cấp độ log
        log_levels = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.NOTICE: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ALERT: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }
        
        # Ghi log với cấp độ phù hợp
        self.logger.log(
            log_levels[alert.level],
            f"[{alert.category.name}] {alert.message} (ID: {alert.alert_id})",
            extra={"alert_data": alert.data}
        )
        
        return True

class FileAlertHandler(AlertHandler):
    """
    Handler lưu cảnh báo vào file.
    """
    
    def __init__(
        self, 
        name: str, 
        min_level: AlertLevel = AlertLevel.INFO,
        file_path: Optional[Union[str, Path]] = None
    ):
        """
        Khởi tạo handler.
        
        Args:
            name: Tên handler
            min_level: Cấp độ cảnh báo tối thiểu để xử lý
            file_path: Đường dẫn file lưu cảnh báo
        """
        super().__init__(name, min_level)
        
        if file_path is None:
            # Sử dụng đường dẫn mặc định
            file_path = Path("logs/alerts") / f"alerts_{datetime.now().strftime('%Y%m%d')}.json"
        
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Tạo file nếu chưa tồn tại
        if not self.file_path.exists():
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([], f)
    
    async def handle(self, alert: Alert) -> bool:
        """
        Xử lý cảnh báo bằng cách lưu vào file.
        
        Args:
            alert: Cảnh báo cần xử lý
            
        Returns:
            True nếu xử lý thành công, False nếu không
        """
        try:
            # Đọc dữ liệu hiện có
            alerts = []
            
            if self.file_path.exists():
                with open(self.file_path, "r", encoding="utf-8") as f:
                    try:
                        alerts = json.load(f)
                    except json.JSONDecodeError:
                        # File rỗng hoặc không phải JSON hợp lệ
                        alerts = []
            
            # Thêm cảnh báo mới
            alerts.append(alert.to_dict())
            
            # Lưu lại vào file
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(alerts, f, ensure_ascii=False, indent=4)
            
            self.logger.debug(f"Đã lưu cảnh báo {alert.alert_id} vào {self.file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu cảnh báo: {str(e)}", exc_info=True)
            return False

class CallbackAlertHandler(AlertHandler):
    """
    Handler gọi callback function.
    """
    
    def __init__(
        self, 
        name: str, 
        callback: Callable[[Alert], None],
        min_level: AlertLevel = AlertLevel.INFO
    ):
        """
        Khởi tạo handler.
        
        Args:
            name: Tên handler
            callback: Hàm callback nhận đối tượng Alert
            min_level: Cấp độ cảnh báo tối thiểu để xử lý
        """
        super().__init__(name, min_level)
        self.callback = callback
    
    async def handle(self, alert: Alert) -> bool:
        """
        Xử lý cảnh báo bằng cách gọi callback.
        
        Args:
            alert: Cảnh báo cần xử lý
            
        Returns:
            True nếu xử lý thành công, False nếu không
        """
        try:
            # Gọi callback
            self.callback(alert)
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi gọi callback: {str(e)}", exc_info=True)
            return False

class AlertSystem:
    """
    Hệ thống cảnh báo trung tâm.
    Quản lý việc tạo, xử lý và theo dõi cảnh báo trong hệ thống.
    """
    
    def __init__(
        self,
        enable_console: bool = True,
        enable_file: bool = True,
        alert_file_path: Optional[Union[str, Path]] = None,
        min_level: AlertLevel = AlertLevel.INFO,
        deduplicate_interval: int = 300,  # 5 phút
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo hệ thống cảnh báo.
        
        Args:
            enable_console: Bật hiển thị cảnh báo trên console
            enable_file: Bật lưu cảnh báo vào file
            alert_file_path: Đường dẫn file lưu cảnh báo
            min_level: Cấp độ cảnh báo tối thiểu
            deduplicate_interval: Khoảng thời gian khử trùng cảnh báo (giây)
            logger: Logger tùy chỉnh
        """
        # Logger
        self.logger = logger or SystemLogger("alert_system")
        
        # Cấu hình
        self.system_config = get_system_config()
        self.min_level = min_level
        self.deduplicate_interval = deduplicate_interval
        
        # Hàng đợi cảnh báo
        self.alert_queue = asyncio.Queue()
        
        # Danh sách các handler
        self.handlers = []
        
        # Tập hợp ID cảnh báo đã xử lý gần đây (để khử trùng)
        self.recent_alert_hashes = set()
        
        # Thread xử lý cảnh báo
        self.processor_task = None
        self.running = False
        
        # Lịch sử cảnh báo
        self.alert_history = []
        self.max_history_size = 1000
        
        # Thêm các handler mặc định
        if enable_console:
            self.add_handler(ConsoleAlertHandler("console", min_level))
        
        if enable_file:
            self.add_handler(FileAlertHandler("file", min_level, alert_file_path))
        
        self.logger.info(f"Khởi tạo AlertSystem với min_level={min_level.name}, deduplicate_interval={deduplicate_interval}s")
    
    def add_handler(self, handler: AlertHandler) -> None:
        """
        Thêm handler mới.
        
        Args:
            handler: Handler cần thêm
        """
        self.handlers.append(handler)
        self.logger.debug(f"Đã thêm handler {handler.name}")
    
    def remove_handler(self, handler_name: str) -> bool:
        """
        Xóa handler.
        
        Args:
            handler_name: Tên handler cần xóa
            
        Returns:
            True nếu xóa thành công, False nếu không tìm thấy
        """
        for i, handler in enumerate(self.handlers):
            if handler.name == handler_name:
                self.handlers.pop(i)
                self.logger.debug(f"Đã xóa handler {handler_name}")
                return True
        
        self.logger.warning(f"Không tìm thấy handler {handler_name}")
        return False
    
    async def start(self) -> bool:
        """
        Bắt đầu hệ thống cảnh báo.
        
        Returns:
            True nếu bắt đầu thành công, False nếu không
        """
        if self.running:
            self.logger.warning("Hệ thống cảnh báo đã đang chạy")
            return False
        
        self.running = True
        self.processor_task = asyncio.create_task(self._process_alerts())
        
        self.logger.info("Đã bắt đầu hệ thống cảnh báo")
        return True
    
    async def stop(self) -> bool:
        """
        Dừng hệ thống cảnh báo.
        
        Returns:
            True nếu dừng thành công, False nếu không
        """
        if not self.running:
            self.logger.warning("Hệ thống cảnh báo không đang chạy")
            return False
        
        self.running = False
        
        # Chờ task kết thúc
        if self.processor_task:
            try:
                self.processor_task.cancel()
                await asyncio.wait_for(asyncio.shield(self.processor_task), timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        self.logger.info("Đã dừng hệ thống cảnh báo")
        return True
    
    async def _process_alerts(self) -> None:
        """
        Vòng lặp xử lý cảnh báo.
        """
        self.logger.info("Bắt đầu vòng lặp xử lý cảnh báo")
        
        while self.running:
            try:
                # Lấy cảnh báo từ hàng đợi
                alert = await self.alert_queue.get()
                
                # Tính toán hash để khử trùng
                alert_hash = self._calculate_alert_hash(alert)
                
                # Kiểm tra xem có phải cảnh báo trùng lặp không
                if alert_hash in self.recent_alert_hashes:
                    self.logger.debug(f"Bỏ qua cảnh báo trùng lặp: {alert.message}")
                    self.alert_queue.task_done()
                    continue
                
                # Thêm hash vào danh sách gần đây
                self.recent_alert_hashes.add(alert_hash)
                
                # Lên lịch xóa hash sau khoảng thời gian khử trùng
                asyncio.create_task(self._remove_alert_hash(alert_hash))
                
                # Thêm vào lịch sử
                self._add_to_history(alert)
                
                # Xử lý cảnh báo với tất cả các handler phù hợp
                for handler in self.handlers:
                    if handler.can_handle(alert):
                        try:
                            await handler.handle(alert)
                        except Exception as e:
                            self.logger.error(f"Lỗi khi xử lý cảnh báo với handler {handler.name}: {str(e)}", exc_info=True)
                
                # Đánh dấu cảnh báo đã được xử lý
                self.alert_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Lỗi trong vòng lặp xử lý cảnh báo: {str(e)}", exc_info=True)
                await asyncio.sleep(1)  # Đợi một chút trước khi thử lại
        
        self.logger.info("Kết thúc vòng lặp xử lý cảnh báo")
    
    async def _remove_alert_hash(self, alert_hash: str) -> None:
        """
        Xóa hash cảnh báo sau khoảng thời gian.
        
        Args:
            alert_hash: Hash cảnh báo cần xóa
        """
        await asyncio.sleep(self.deduplicate_interval)
        self.recent_alert_hashes.discard(alert_hash)
    
    def _calculate_alert_hash(self, alert: Alert) -> str:
        """
        Tính toán hash cho cảnh báo để khử trùng.
        
        Args:
            alert: Cảnh báo cần tính hash
            
        Returns:
            Hash cảnh báo
        """
        import hashlib
        
        # Tạo chuỗi key từ các thông tin cơ bản của cảnh báo
        key = f"{alert.message}:{alert.level.name}:{alert.category.name}:{alert.source}"
        
        # Băm chuỗi key
        return hashlib.md5(key.encode()).hexdigest()
    
    def _add_to_history(self, alert: Alert) -> None:
        """
        Thêm cảnh báo vào lịch sử.
        
        Args:
            alert: Cảnh báo cần thêm
        """
        self.alert_history.append(alert)
        
        # Giới hạn kích thước lịch sử
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]
    
    async def alert(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        category: AlertCategory = AlertCategory.SYSTEM,
        source: str = "system",
        data: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        Tạo và xử lý cảnh báo mới.
        
        Args:
            message: Thông điệp cảnh báo
            level: Cấp độ cảnh báo
            category: Loại cảnh báo
            source: Nguồn phát sinh cảnh báo
            data: Dữ liệu bổ sung
            
        Returns:
            Đối tượng Alert đã tạo
        """
        # Kiểm tra cấp độ tối thiểu
        if level.value < self.min_level.value:
            return None
        
        # Tạo cảnh báo mới
        alert = Alert(
            message=message,
            level=level,
            category=category,
            source=source,
            data=data
        )
        
        # Thêm vào hàng đợi xử lý
        await self.alert_queue.put(alert)
        
        return alert
    
    def info(self, message: str, category: AlertCategory = AlertCategory.SYSTEM, 
            source: str = "system", data: Optional[Dict[str, Any]] = None) -> None:
        """
        Tạo cảnh báo mức INFO.
        
        Args:
            message: Thông điệp cảnh báo
            category: Loại cảnh báo
            source: Nguồn phát sinh cảnh báo
            data: Dữ liệu bổ sung
        """
        asyncio.create_task(self.alert(
            message=message,
            level=AlertLevel.INFO,
            category=category,
            source=source,
            data=data
        ))
    
    def notice(self, message: str, category: AlertCategory = AlertCategory.SYSTEM, 
              source: str = "system", data: Optional[Dict[str, Any]] = None) -> None:
        """
        Tạo cảnh báo mức NOTICE.
        
        Args:
            message: Thông điệp cảnh báo
            category: Loại cảnh báo
            source: Nguồn phát sinh cảnh báo
            data: Dữ liệu bổ sung
        """
        asyncio.create_task(self.alert(
            message=message,
            level=AlertLevel.NOTICE,
            category=category,
            source=source,
            data=data
        ))
    
    def warning(self, message: str, category: AlertCategory = AlertCategory.SYSTEM, 
               source: str = "system", data: Optional[Dict[str, Any]] = None) -> None:
        """
        Tạo cảnh báo mức WARNING.
        
        Args:
            message: Thông điệp cảnh báo
            category: Loại cảnh báo
            source: Nguồn phát sinh cảnh báo
            data: Dữ liệu bổ sung
        """
        asyncio.create_task(self.alert(
            message=message,
            level=AlertLevel.WARNING,
            category=category,
            source=source,
            data=data
        ))
    
    def alert_msg(self, message: str, category: AlertCategory = AlertCategory.SYSTEM, 
                source: str = "system", data: Optional[Dict[str, Any]] = None) -> None:
        """
        Tạo cảnh báo mức ALERT.
        
        Args:
            message: Thông điệp cảnh báo
            category: Loại cảnh báo
            source: Nguồn phát sinh cảnh báo
            data: Dữ liệu bổ sung
        """
        asyncio.create_task(self.alert(
            message=message,
            level=AlertLevel.ALERT,
            category=category,
            source=source,
            data=data
        ))
    
    def critical(self, message: str, category: AlertCategory = AlertCategory.SYSTEM, 
                source: str = "system", data: Optional[Dict[str, Any]] = None) -> None:
        """
        Tạo cảnh báo mức CRITICAL.
        
        Args:
            message: Thông điệp cảnh báo
            category: Loại cảnh báo
            source: Nguồn phát sinh cảnh báo
            data: Dữ liệu bổ sung
        """
        asyncio.create_task(self.alert(
            message=message,
            level=AlertLevel.CRITICAL,
            category=category,
            source=source,
            data=data
        ))
    
    def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """
        Lấy cảnh báo theo ID.
        
        Args:
            alert_id: ID cảnh báo cần tìm
            
        Returns:
            Đối tượng Alert hoặc None nếu không tìm thấy
        """
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                return alert
        return None
    
    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        category: Optional[AlertCategory] = None,
        source: Optional[str] = None,
        resolved: Optional[bool] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Alert]:
        """
        Lấy danh sách cảnh báo theo điều kiện.
        
        Args:
            level: Lọc theo cấp độ
            category: Lọc theo loại
            source: Lọc theo nguồn
            resolved: Lọc theo trạng thái giải quyết
            start_time: Thời gian bắt đầu
            end_time: Thời gian kết thúc
            limit: Số lượng tối đa kết quả
            
        Returns:
            Danh sách các cảnh báo thỏa mãn điều kiện
        """
        filtered_alerts = []
        
        for alert in reversed(self.alert_history):  # Duyệt từ mới đến cũ
            # Kiểm tra các điều kiện lọc
            if level is not None and alert.level != level:
                continue
            
            if category is not None and alert.category != category:
                continue
            
            if source is not None and alert.source != source:
                continue
            
            if resolved is not None and alert.resolved != resolved:
                continue
            
            if start_time is not None and alert.timestamp < start_time:
                continue
            
            if end_time is not None and alert.timestamp > end_time:
                continue
            
            # Thêm vào kết quả
            filtered_alerts.append(alert)
            
            # Kiểm tra giới hạn
            if len(filtered_alerts) >= limit:
                break
        
        return filtered_alerts
    
    def acknowledge_alert(self, alert_id: str, by: Optional[str] = None) -> bool:
        """
        Xác nhận đã biết về cảnh báo.
        
        Args:
            alert_id: ID cảnh báo cần xác nhận
            by: Người xác nhận
            
        Returns:
            True nếu xác nhận thành công, False nếu không
        """
        alert = self.get_alert_by_id(alert_id)
        
        if alert is None:
            self.logger.warning(f"Không tìm thấy cảnh báo {alert_id} để xác nhận")
            return False
        
        alert.acknowledge(by)
        self.logger.info(f"Đã xác nhận cảnh báo {alert_id}" + (f" bởi {by}" if by else ""))
        return True
    
    def resolve_alert(self, alert_id: str, by: Optional[str] = None) -> bool:
        """
        Đánh dấu cảnh báo đã được giải quyết.
        
        Args:
            alert_id: ID cảnh báo cần giải quyết
            by: Người giải quyết
            
        Returns:
            True nếu giải quyết thành công, False nếu không
        """
        alert = self.get_alert_by_id(alert_id)
        
        if alert is None:
            self.logger.warning(f"Không tìm thấy cảnh báo {alert_id} để giải quyết")
            return False
        
        alert.resolve(by)
        self.logger.info(f"Đã giải quyết cảnh báo {alert_id}" + (f" bởi {by}" if by else ""))
        return True
    
    def add_note_to_alert(self, alert_id: str, note: str, by: Optional[str] = None) -> bool:
        """
        Thêm ghi chú cho cảnh báo.
        
        Args:
            alert_id: ID cảnh báo cần thêm ghi chú
            note: Nội dung ghi chú
            by: Người ghi chú
            
        Returns:
            True nếu thêm thành công, False nếu không
        """
        alert = self.get_alert_by_id(alert_id)
        
        if alert is None:
            self.logger.warning(f"Không tìm thấy cảnh báo {alert_id} để thêm ghi chú")
            return False
        
        alert.add_note(note, by)
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê về cảnh báo.
        
        Returns:
            Dict chứa thống kê cảnh báo
        """
        # Đếm số lượng cảnh báo theo cấp độ
        level_counts = {}
        for level in AlertLevel:
            level_counts[level.name] = 0
        
        # Đếm số lượng cảnh báo theo loại
        category_counts = {}
        for category in AlertCategory:
            category_counts[category.name] = 0
        
        # Đếm số lượng cảnh báo đã giải quyết và chưa giải quyết
        resolved_count = 0
        unresolved_count = 0
        
        # Đếm số lượng cảnh báo theo thời gian
        last_hour_count = 0
        last_day_count = 0
        
        # Thời gian hiện tại
        now = datetime.now()
        
        for alert in self.alert_history:
            # Đếm theo cấp độ
            level_counts[alert.level.name] += 1
            
            # Đếm theo loại
            category_counts[alert.category.name] += 1
            
            # Đếm theo trạng thái giải quyết
            if alert.resolved:
                resolved_count += 1
            else:
                unresolved_count += 1
            
            # Đếm theo thời gian
            if now - alert.timestamp < timedelta(hours=1):
                last_hour_count += 1
            
            if now - alert.timestamp < timedelta(days=1):
                last_day_count += 1
        
        return {
            "total_alerts": len(self.alert_history),
            "levels": level_counts,
            "categories": category_counts,
            "resolved": resolved_count,
            "unresolved": unresolved_count,
            "last_hour": last_hour_count,
            "last_day": last_day_count,
            "last_alert_time": self.alert_history[-1].timestamp.isoformat() if self.alert_history else None
        }
    
    def export_alerts(self, file_path: Union[str, Path]) -> bool:
        """
        Xuất danh sách cảnh báo ra file.
        
        Args:
            file_path: Đường dẫn file
            
        Returns:
            True nếu xuất thành công, False nếu không
        """
        try:
            # Chuyển đổi các đối tượng Alert thành dict
            alerts_dict = [alert.to_dict() for alert in self.alert_history]
            
            # Đảm bảo thư mục tồn tại
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Lưu vào file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(alerts_dict, f, ensure_ascii=False, indent=4)
            
            self.logger.info(f"Đã xuất {len(alerts_dict)} cảnh báo vào {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi xuất cảnh báo: {str(e)}", exc_info=True)
            return False
    
    def import_alerts(self, file_path: Union[str, Path]) -> bool:
        """
        Nhập danh sách cảnh báo từ file.
        
        Args:
            file_path: Đường dẫn file
            
        Returns:
            True nếu nhập thành công, False nếu không
        """
        try:
            # Đọc dữ liệu từ file
            with open(file_path, "r", encoding="utf-8") as f:
                alerts_dict = json.load(f)
            
            # Chuyển đổi từ dict thành đối tượng Alert
            imported_alerts = []
            for alert_dict in alerts_dict:
                try:
                    alert = Alert.from_dict(alert_dict)
                    imported_alerts.append(alert)
                except Exception as e:
                    self.logger.warning(f"Lỗi khi chuyển đổi cảnh báo: {str(e)}")
            
            # Thêm vào lịch sử
            self.alert_history.extend(imported_alerts)
            
            # Giới hạn kích thước lịch sử
            if len(self.alert_history) > self.max_history_size:
                self.alert_history = self.alert_history[-self.max_history_size:]
            
            self.logger.info(f"Đã nhập {len(imported_alerts)} cảnh báo từ {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi nhập cảnh báo: {str(e)}", exc_info=True)
            return False
    
    def clear_history(self) -> None:
        """
        Xóa lịch sử cảnh báo.
        """
        self.alert_history = []
        self.logger.info("Đã xóa lịch sử cảnh báo")
    
    def set_min_level(self, level: AlertLevel) -> None:
        """
        Đặt cấp độ cảnh báo tối thiểu.
        
        Args:
            level: Cấp độ cảnh báo tối thiểu mới
        """
        self.min_level = level
        self.logger.info(f"Đã đặt cấp độ cảnh báo tối thiểu thành {level.name}")

# Singleton instance
_alert_system_instance = None

def get_alert_system() -> AlertSystem:
    """
    Lấy instance singleton của AlertSystem.
    
    Returns:
        Instance AlertSystem
    """
    global _alert_system_instance
    if _alert_system_instance is None:
        _alert_system_instance = AlertSystem()
    return _alert_system_instance