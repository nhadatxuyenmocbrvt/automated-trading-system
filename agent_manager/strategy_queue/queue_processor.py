"""
Bộ xử lý hàng đợi chiến lược.
File này định nghĩa lớp QueueProcessor để quản lý và xử lý
hàng đợi chiến lược, theo dõi vòng đời của các chiến lược,
và điều phối việc thực thi chúng theo thứ tự ưu tiên.
"""

import time
import logging
import threading
import queue
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from enum import Enum
from datetime import datetime
import json
from pathlib import Path
import uuid

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import get_logger
from config.system_config import get_system_config, MODEL_DIR
from agent_manager.strategy_queue.strategy_manager import StrategyManager, StrategyStatus
from agent_manager.strategy_queue.strategy_selector import StrategySelector

class QueueItemStatus(Enum):
    """
    Trạng thái của một mục trong hàng đợi.
    Định nghĩa các trạng thái có thể có của một mục trong hàng đợi.
    """
    PENDING = "pending"           # Đang chờ xử lý
    PROCESSING = "processing"     # Đang được xử lý
    COMPLETED = "completed"       # Đã hoàn thành
    FAILED = "failed"             # Thất bại
    PAUSED = "paused"             # Tạm dừng
    CANCELLED = "cancelled"       # Đã hủy
    TIMEOUT = "timeout"           # Hết thời gian
    RETRY = "retry"               # Đang thử lại

class QueueProcessor:
    """
    Lớp xử lý hàng đợi chiến lược.
    Quản lý hàng đợi của các chiến lược đã được lựa chọn, điều phối
    việc thực thi chiến lược và theo dõi trạng thái của chúng.
    """
    
    def __init__(
        self,
        strategy_manager: StrategyManager,
        strategy_selector: StrategySelector,
        max_queue_size: int = 100,
        processing_interval: float = 1.0,
        auto_start: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        name: str = "queue_processor",
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Khởi tạo bộ xử lý hàng đợi.
        
        Args:
            strategy_manager: Quản lý chiến lược
            strategy_selector: Bộ chọn chiến lược
            max_queue_size: Kích thước tối đa của hàng đợi
            processing_interval: Khoảng thời gian giữa các lần xử lý (giây)
            auto_start: Tự động bắt đầu xử lý hàng đợi khi khởi tạo
            save_dir: Thư mục lưu trạng thái hàng đợi
            name: Tên của bộ xử lý hàng đợi
            logger: Logger tùy chỉnh
            **kwargs: Các tham số tùy chọn khác
        """
        # Thiết lập logger
        self.logger = logger or get_logger("queue_processor")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập các thuộc tính
        self.strategy_manager = strategy_manager
        self.strategy_selector = strategy_selector
        self.max_queue_size = max_queue_size
        self.processing_interval = processing_interval
        self.name = name
        self.kwargs = kwargs
        
        # Thư mục lưu trữ
        if save_dir is None:
            save_dir = MODEL_DIR / 'strategy_queue'
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo file name.json để lưu
        self.save_file = self.save_dir / f"{self.name}_queue.json"
        
        # Khởi tạo hàng đợi
        self.queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.queue_items = {}  # Lưu trữ thông tin về các mục trong hàng đợi (id: thông tin)
        
        # Bộ đếm cho ưu tiên (để tránh việc phá vỡ nguyên tắc FIFO khi có cùng ưu tiên)
        self.priority_counter = 0
        
        # Thread xử lý hàng đợi
        self.processing_thread = None
        self.is_running = False
        self.stop_event = threading.Event()
        
        # Các biến theo dõi
        self.processed_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.last_processed_time = 0
        
        # Callback functions
        self.on_item_complete = kwargs.get("on_item_complete", None)
        self.on_item_fail = kwargs.get("on_item_fail", None)
        self.execute_strategy_fn = kwargs.get("execute_strategy_fn", None)
        
        # Tải trạng thái hàng đợi nếu có
        if self.save_file.exists():
            self.load_queue()
        
        # Tự động bắt đầu nếu được cấu hình
        if auto_start:
            self.start()
        
        self.logger.info(
            f"Đã khởi tạo {self.__class__.__name__} với max_queue_size={max_queue_size}, "
            f"processing_interval={processing_interval}, auto_start={auto_start}"
        )
    
    def start(self) -> bool:
        """
        Bắt đầu xử lý hàng đợi.
        
        Returns:
            True nếu bắt đầu thành công, False nếu đã đang chạy
        """
        if self.is_running:
            self.logger.warning("Bộ xử lý hàng đợi đã đang chạy")
            return False
        
        # Đặt lại sự kiện dừng
        self.stop_event.clear()
        
        # Khởi tạo và bắt đầu thread xử lý
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name=f"{self.name}_processing_thread",
            daemon=True
        )
        
        self.is_running = True
        self.processing_thread.start()
        
        self.logger.info("Đã bắt đầu xử lý hàng đợi")
        return True
    
    def stop(self) -> bool:
        """
        Dừng xử lý hàng đợi.
        
        Returns:
            True nếu dừng thành công, False nếu đã dừng
        """
        if not self.is_running:
            self.logger.warning("Bộ xử lý hàng đợi đã dừng")
            return False
        
        # Đặt sự kiện dừng
        self.stop_event.set()
        
        # Chờ thread kết thúc với timeout
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        self.is_running = False
        self.logger.info("Đã dừng xử lý hàng đợi")
        return True
    
    def add_strategy_to_queue(
        self,
        strategy_id: str,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        timeout: float = 300.0  # 5 phút
    ) -> Tuple[bool, str]:
        """
        Thêm chiến lược vào hàng đợi.
        
        Args:
            strategy_id: ID của chiến lược
            priority: Ưu tiên của chiến lược (số càng cao càng ưu tiên)
            metadata: Thông tin bổ sung
            max_retries: Số lần thử lại tối đa
            timeout: Thời gian tối đa được phép xử lý (giây)
            
        Returns:
            Tuple (success, queue_item_id)
        """
        # Kiểm tra xem chiến lược tồn tại không
        strategy = self.strategy_manager.get_strategy(strategy_id)
        if not strategy:
            self.logger.warning(f"Không tìm thấy chiến lược với ID: {strategy_id}")
            return False, ""
        
        # Tạo ID cho mục hàng đợi
        queue_item_id = str(uuid.uuid4())
        
        # Tăng bộ đếm ưu tiên
        self.priority_counter += 1
        
        # Chuẩn bị thông tin mục hàng đợi
        queue_item = {
            "id": queue_item_id,
            "strategy_id": strategy_id,
            "priority": priority,
            "status": QueueItemStatus.PENDING.value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "start_time": None,
            "end_time": None,
            "metadata": metadata or {},
            "result": None,
            "error": None,
            "retry_count": 0,
            "max_retries": max_retries,
            "timeout": timeout
        }
        
        # Lưu thông tin mục
        self.queue_items[queue_item_id] = queue_item
        
        try:
            # Thêm vào hàng đợi với ưu tiên
            # Sử dụng negative priority để PriorityQueue lấy cao nhất trước
            # Kết hợp priority_counter để duy trì thứ tự FIFO cho các mục cùng ưu tiên
            self.queue.put((-priority, self.priority_counter, queue_item_id), block=False)
            self.logger.info(f"Đã thêm chiến lược {strategy_id} vào hàng đợi với ID: {queue_item_id}")
            return True, queue_item_id
        except queue.Full:
            self.logger.error("Hàng đợi đã đầy, không thể thêm chiến lược")
            del self.queue_items[queue_item_id]
            return False, queue_item_id
    
    def add_strategies_from_selector(
        self,
        market_context: Optional[Dict[str, Any]] = None,
        count: Optional[int] = None,
        filter_params: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Thêm các chiến lược được chọn bởi bộ chọn vào hàng đợi.
        
        Args:
            market_context: Thông tin ngữ cảnh thị trường
            count: Số lượng chiến lược cần chọn
            filter_params: Tham số lọc chiến lược
            priority: Ưu tiên của các chiến lược
            metadata: Thông tin bổ sung
            
        Returns:
            Danh sách ID của các mục hàng đợi đã thêm
        """
        # Lựa chọn chiến lược
        selected_strategies = self.strategy_selector.select_strategies(
            market_context=market_context,
            count=count,
            filter_params=filter_params
        )
        
        if not selected_strategies:
            self.logger.warning("Không có chiến lược nào được chọn để thêm vào hàng đợi")
            return []
        
        # Thêm các chiến lược vào hàng đợi
        added_items = []
        
        for strategy_id in selected_strategies:
            success, queue_item_id = self.add_strategy_to_queue(
                strategy_id=strategy_id,
                priority=priority,
                metadata=metadata
            )
            
            if success:
                added_items.append(queue_item_id)
        
        self.logger.info(f"Đã thêm {len(added_items)} chiến lược vào hàng đợi từ bộ chọn")
        return added_items
    
    def get_queue_item(self, queue_item_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin về mục trong hàng đợi.
        
        Args:
            queue_item_id: ID của mục hàng đợi
            
        Returns:
            Thông tin mục hàng đợi nếu tồn tại, None nếu không
        """
        if queue_item_id not in self.queue_items:
            self.logger.warning(f"Không tìm thấy mục hàng đợi với ID: {queue_item_id}")
            return None
        
        return self.queue_items[queue_item_id].copy()
    
    def get_all_queue_items(self) -> Dict[str, Dict[str, Any]]:
        """
        Lấy thông tin về tất cả mục trong hàng đợi.
        
        Returns:
            Dict chứa thông tin về tất cả mục hàng đợi
        """
        return self.queue_items.copy()
    
    def get_queue_items_by_status(
        self,
        status: Union[str, QueueItemStatus]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Lấy thông tin về các mục trong hàng đợi theo trạng thái.
        
        Args:
            status: Trạng thái cần lọc
            
        Returns:
            Dict chứa thông tin về các mục hàng đợi có trạng thái tương ứng
        """
        if isinstance(status, QueueItemStatus):
            status = status.value
        
        filtered_items = {}
        
        for item_id, item in self.queue_items.items():
            if item.get("status") == status:
                filtered_items[item_id] = item.copy()
        
        return filtered_items
    
    def update_queue_item_status(
        self,
        queue_item_id: str,
        status: Union[str, QueueItemStatus],
        result: Any = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Cập nhật trạng thái của mục trong hàng đợi.
        
        Args:
            queue_item_id: ID của mục hàng đợi
            status: Trạng thái mới
            result: Kết quả xử lý
            error: Thông tin lỗi
            
        Returns:
            True nếu cập nhật thành công, False nếu không
        """
        if queue_item_id not in self.queue_items:
            self.logger.warning(f"Không tìm thấy mục hàng đợi với ID: {queue_item_id}")
            return False
        
        if isinstance(status, QueueItemStatus):
            status = status.value
        
        # Cập nhật trạng thái
        self.queue_items[queue_item_id]["status"] = status
        self.queue_items[queue_item_id]["updated_at"] = datetime.now().isoformat()
        
        # Cập nhật kết quả và lỗi nếu có
        if result is not None:
            self.queue_items[queue_item_id]["result"] = result
        
        if error is not None:
            self.queue_items[queue_item_id]["error"] = error
        
        # Cập nhật thời gian kết thúc nếu đã hoàn thành hoặc thất bại
        if status in [QueueItemStatus.COMPLETED.value, QueueItemStatus.FAILED.value, 
                     QueueItemStatus.CANCELLED.value, QueueItemStatus.TIMEOUT.value]:
            self.queue_items[queue_item_id]["end_time"] = datetime.now().isoformat()
        
        # Đếm số lần thành công và thất bại
        if status == QueueItemStatus.COMPLETED.value:
            self.success_count += 1
        elif status == QueueItemStatus.FAILED.value:
            self.failure_count += 1
        
        # Gọi callback nếu có
        if status == QueueItemStatus.COMPLETED.value and self.on_item_complete:
            try:
                self.on_item_complete(self.queue_items[queue_item_id])
            except Exception as e:
                self.logger.error(f"Lỗi khi gọi on_item_complete: {str(e)}")
        
        if status == QueueItemStatus.FAILED.value and self.on_item_fail:
            try:
                self.on_item_fail(self.queue_items[queue_item_id])
            except Exception as e:
                self.logger.error(f"Lỗi khi gọi on_item_fail: {str(e)}")
        
        self.logger.info(f"Đã cập nhật trạng thái mục hàng đợi {queue_item_id} thành {status}")
        return True
    
    def remove_queue_item(self, queue_item_id: str) -> bool:
        """
        Xóa mục khỏi hàng đợi.
        
        Args:
            queue_item_id: ID của mục hàng đợi
            
        Returns:
            True nếu xóa thành công, False nếu không
        """
        if queue_item_id not in self.queue_items:
            self.logger.warning(f"Không tìm thấy mục hàng đợi với ID: {queue_item_id}")
            return False
        
        # Kiểm tra trạng thái
        status = self.queue_items[queue_item_id]["status"]
        if status == QueueItemStatus.PROCESSING.value:
            self.logger.warning(f"Không thể xóa mục hàng đợi {queue_item_id} đang được xử lý")
            return False
        
        # Xóa khỏi dict
        del self.queue_items[queue_item_id]
        
        self.logger.info(f"Đã xóa mục hàng đợi {queue_item_id}")
        return True
    
    def retry_failed_item(self, queue_item_id: str) -> bool:
        """
        Thử lại mục hàng đợi đã thất bại.
        
        Args:
            queue_item_id: ID của mục hàng đợi
            
        Returns:
            True nếu thử lại thành công, False nếu không
        """
        if queue_item_id not in self.queue_items:
            self.logger.warning(f"Không tìm thấy mục hàng đợi với ID: {queue_item_id}")
            return False
        
        # Kiểm tra trạng thái
        item = self.queue_items[queue_item_id]
        if item["status"] not in [QueueItemStatus.FAILED.value, QueueItemStatus.TIMEOUT.value]:
            self.logger.warning(f"Chỉ có thể thử lại mục hàng đợi đã thất bại hoặc hết thời gian")
            return False
        
        # Kiểm tra số lần thử lại
        if item["retry_count"] >= item["max_retries"]:
            self.logger.warning(f"Đã vượt quá số lần thử lại tối đa ({item['max_retries']})")
            return False
        
        # Cập nhật thông tin mục
        item["retry_count"] += 1
        item["status"] = QueueItemStatus.RETRY.value
        item["updated_at"] = datetime.now().isoformat()
        item["start_time"] = None
        item["end_time"] = None
        item["result"] = None
        item["error"] = None
        
        try:
            # Thêm lại vào hàng đợi
            self.queue.put((-item["priority"], self.priority_counter, queue_item_id), block=False)
            self.priority_counter += 1
            
            self.logger.info(f"Đã thêm lại mục hàng đợi {queue_item_id} để thử lại (lần {item['retry_count']})")
            return True
        except queue.Full:
            self.logger.error("Hàng đợi đã đầy, không thể thêm lại mục")
            return False
    
    def pause_queue_item(self, queue_item_id: str) -> bool:
        """
        Tạm dừng mục hàng đợi.
        
        Args:
            queue_item_id: ID của mục hàng đợi
            
        Returns:
            True nếu tạm dừng thành công, False nếu không
        """
        if queue_item_id not in self.queue_items:
            self.logger.warning(f"Không tìm thấy mục hàng đợi với ID: {queue_item_id}")
            return False
        
        # Kiểm tra trạng thái
        item = self.queue_items[queue_item_id]
        if item["status"] == QueueItemStatus.PROCESSING.value:
            # Không thể tạm dừng mục đang xử lý
            self.logger.warning(f"Không thể tạm dừng mục hàng đợi {queue_item_id} đang được xử lý")
            return False
        
        # Cập nhật trạng thái
        return self.update_queue_item_status(queue_item_id, QueueItemStatus.PAUSED)
    
    def resume_queue_item(self, queue_item_id: str) -> bool:
        """
        Tiếp tục mục hàng đợi đã tạm dừng.
        
        Args:
            queue_item_id: ID của mục hàng đợi
            
        Returns:
            True nếu tiếp tục thành công, False nếu không
        """
        if queue_item_id not in self.queue_items:
            self.logger.warning(f"Không tìm thấy mục hàng đợi với ID: {queue_item_id}")
            return False
        
        # Kiểm tra trạng thái
        item = self.queue_items[queue_item_id]
        if item["status"] != QueueItemStatus.PAUSED.value:
            self.logger.warning(f"Chỉ có thể tiếp tục mục hàng đợi đã tạm dừng")
            return False
        
        # Cập nhật trạng thái
        item["status"] = QueueItemStatus.PENDING.value
        item["updated_at"] = datetime.now().isoformat()
        
        try:
            # Thêm lại vào hàng đợi
            self.queue.put((-item["priority"], self.priority_counter, queue_item_id), block=False)
            self.priority_counter += 1
            
            self.logger.info(f"Đã tiếp tục mục hàng đợi {queue_item_id}")
            return True
        except queue.Full:
            self.logger.error("Hàng đợi đã đầy, không thể tiếp tục mục")
            return False
    
    def cancel_queue_item(self, queue_item_id: str) -> bool:
        """
        Hủy bỏ mục hàng đợi.
        
        Args:
            queue_item_id: ID của mục hàng đợi
            
        Returns:
            True nếu hủy bỏ thành công, False nếu không
        """
        if queue_item_id not in self.queue_items:
            self.logger.warning(f"Không tìm thấy mục hàng đợi với ID: {queue_item_id}")
            return False
        
        # Kiểm tra trạng thái
        item = self.queue_items[queue_item_id]
        if item["status"] == QueueItemStatus.PROCESSING.value:
            # Không thể hủy bỏ mục đang xử lý
            self.logger.warning(f"Không thể hủy bỏ mục hàng đợi {queue_item_id} đang được xử lý")
            return False
        
        # Cập nhật trạng thái
        return self.update_queue_item_status(queue_item_id, QueueItemStatus.CANCELLED)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê về hàng đợi.
        
        Returns:
            Dict chứa thống kê
        """
        # Đếm các trạng thái
        status_counts = {}
        for item in self.queue_items.values():
            status = item.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Tính toán thời gian xử lý trung bình
        processing_times = []
        
        for item in self.queue_items.values():
            if item.get("start_time") and item.get("end_time"):
                try:
                    start_time = datetime.fromisoformat(item["start_time"])
                    end_time = datetime.fromisoformat(item["end_time"])
                    processing_time = (end_time - start_time).total_seconds()
                    processing_times.append(processing_time)
                except (ValueError, TypeError):
                    pass
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "total_items": len(self.queue_items),
            "status_counts": status_counts,
            "processed_count": self.processed_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "avg_processing_time": avg_processing_time,
            "is_running": self.is_running,
            "approximate_queue_size": self.queue.qsize()
        }
    
    def clear_completed_items(self, keep_last_n: int = 10) -> int:
        """
        Xóa các mục đã hoàn thành khỏi danh sách (trừ n mục gần nhất).
        
        Args:
            keep_last_n: Số lượng mục gần nhất cần giữ lại
            
        Returns:
            Số lượng mục đã xóa
        """
        # Lọc các mục đã hoàn thành
        completed_items = []
        
        for item_id, item in self.queue_items.items():
            if item["status"] in [QueueItemStatus.COMPLETED.value, QueueItemStatus.CANCELLED.value]:
                try:
                    updated_at = datetime.fromisoformat(item["updated_at"])
                    completed_items.append((item_id, updated_at))
                except (ValueError, TypeError):
                    # Nếu không phân tích được thời gian, thêm vào cuối
                    completed_items.append((item_id, datetime.min))
        
        # Sắp xếp theo thời gian cập nhật
        completed_items.sort(key=lambda x: x[1], reverse=True)
        
        # Giữ lại n mục gần nhất
        items_to_remove = completed_items[keep_last_n:]
        
        # Xóa các mục
        removed_count = 0
        for item_id, _ in items_to_remove:
            if self.remove_queue_item(item_id):
                removed_count += 1
        
        if removed_count > 0:
            self.logger.info(f"Đã xóa {removed_count} mục đã hoàn thành khỏi danh sách")
        
        return removed_count
    
    def save_queue(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Lưu trạng thái hàng đợi vào file.
        
        Args:
            file_path: Đường dẫn file lưu (sử dụng file mặc định nếu không có)
            
        Returns:
            True nếu lưu thành công, False nếu không
        """
        if file_path is None:
            file_path = self.save_file
        
        try:
            # Chuẩn bị dữ liệu lưu
            save_data = {
                "name": self.name,
                "queue_items": self.queue_items,
                "processed_count": self.processed_count,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "last_processed_time": self.last_processed_time,
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "item_count": len(self.queue_items)
                }
            }
            
            # Lưu vào file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Đã lưu trạng thái hàng đợi vào {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu trạng thái hàng đợi: {str(e)}")
            return False
    
    def load_queue(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Tải trạng thái hàng đợi từ file.
        
        Args:
            file_path: Đường dẫn file tải (sử dụng file mặc định nếu không có)
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        if file_path is None:
            file_path = self.save_file
        
        if not os.path.exists(file_path):
            self.logger.warning(f"Không tìm thấy file trạng thái hàng đợi tại {file_path}")
            return False
        
        try:
            # Đọc từ file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Kiểm tra dữ liệu hợp lệ
            if "queue_items" not in data:
                self.logger.error(f"Dữ liệu không hợp lệ trong file {file_path}")
                return False
            
            # Xóa hàng đợi hiện tại
            with self.queue.mutex:
                self.queue.queue.clear()
            
            # Cập nhật dữ liệu
            self.queue_items = data["queue_items"]
            self.processed_count = data.get("processed_count", 0)
            self.success_count = data.get("success_count", 0)
            self.failure_count = data.get("failure_count", 0)
            self.last_processed_time = data.get("last_processed_time", 0)
            
            # Thêm các mục đang chờ xử lý vào hàng đợi
            self.priority_counter = 0
            for item_id, item in self.queue_items.items():
                if item["status"] in [QueueItemStatus.PENDING.value, QueueItemStatus.RETRY.value]:
                    try:
                        self.priority_counter += 1
                        self.queue.put((-item["priority"], self.priority_counter, item_id), block=False)
                    except queue.Full:
                        self.logger.warning(f"Hàng đợi đã đầy, không thể thêm mục {item_id}")
            
            self.logger.info(f"Đã tải trạng thái hàng đợi từ {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tải trạng thái hàng đợi: {str(e)}")
            return False
    
    def _processing_loop(self) -> None:
        """
        Vòng lặp xử lý hàng đợi.
        """
        self.logger.info("Bắt đầu vòng lặp xử lý hàng đợi")
        
        while not self.stop_event.is_set():
            try:
                # Lấy mục từ hàng đợi với timeout
                try:
                    # Định dạng là (negative_priority, counter, queue_item_id)
                    _, _, queue_item_id = self.queue.get(timeout=self.processing_interval)
                except queue.Empty:
                    # Không có mục nào trong hàng đợi
                    continue
                
                # Kiểm tra xem mục có tồn tại không
                if queue_item_id not in self.queue_items:
                    self.logger.warning(f"Mục hàng đợi {queue_item_id} không tồn tại trong danh sách")
                    self.queue.task_done()
                    continue
                
                # Lấy thông tin mục
                item = self.queue_items[queue_item_id]
                
                # Kiểm tra trạng thái
                if item["status"] in [QueueItemStatus.COMPLETED.value, QueueItemStatus.FAILED.value,
                                    QueueItemStatus.CANCELLED.value, QueueItemStatus.TIMEOUT.value]:
                    self.logger.warning(f"Mục hàng đợi {queue_item_id} có trạng thái {item['status']}, bỏ qua")
                    self.queue.task_done()
                    continue
                
                # Cập nhật trạng thái và thời gian bắt đầu
                item["status"] = QueueItemStatus.PROCESSING.value
                item["start_time"] = datetime.now().isoformat()
                item["updated_at"] = datetime.now().isoformat()
                
                # Lấy chiến lược
                strategy_id = item["strategy_id"]
                strategy = self.strategy_manager.get_strategy(strategy_id)
                
                if not strategy:
                    self.logger.error(f"Không tìm thấy chiến lược {strategy_id} cho mục {queue_item_id}")
                    self.update_queue_item_status(
                        queue_item_id,
                        QueueItemStatus.FAILED,
                        error="Chiến lược không tồn tại"
                    )
                    self.queue.task_done()
                    continue
                
                # Gọi hàm thực thi chiến lược
                self.logger.info(f"Bắt đầu xử lý mục hàng đợi {queue_item_id} (chiến lược {strategy_id})")
                
                success = False
                result = None
                error = None
                
                try:
                    # Đặt thời gian bắt đầu
                    start_time = time.time()
                    
                    # Nếu có hàm thực thi tùy chỉnh, sử dụng nó
                    if self.execute_strategy_fn is not None:
                        result = self.execute_strategy_fn(strategy, item)
                        success = True
                    else:
                        # Xử lý mặc định
                        result = self._execute_strategy(strategy, item)
                        success = True
                    
                    # Kiểm tra timeout
                    elapsed_time = time.time() - start_time
                    if elapsed_time > item["timeout"]:
                        self.logger.warning(f"Mục hàng đợi {queue_item_id} đã vượt quá thời gian timeout ({item['timeout']}s)")
                        success = False
                        error = f"Timeout sau {elapsed_time:.2f}s"
                        self.update_queue_item_status(queue_item_id, QueueItemStatus.TIMEOUT, error=error)
                    else:
                        # Cập nhật hiệu suất chiến lược
                        if isinstance(result, dict) and success:
                            self.strategy_manager.update_strategy_performance(
                                strategy_id,
                                result
                            )
                except Exception as e:
                    success = False
                    error = str(e)
                    self.logger.error(f"Lỗi khi xử lý mục hàng đợi {queue_item_id}: {error}")
                
                # Cập nhật trạng thái
                if success:
                    self.update_queue_item_status(queue_item_id, QueueItemStatus.COMPLETED, result=result)
                else:
                    self.update_queue_item_status(queue_item_id, QueueItemStatus.FAILED, error=error)
                    
                    # Kiểm tra nếu cần thử lại
                    if item["retry_count"] < item["max_retries"]:
                        self.logger.info(f"Sẽ thử lại mục hàng đợi {queue_item_id} ({item['retry_count'] + 1}/{item['max_retries']})")
                        self.retry_failed_item(queue_item_id)
                
                # Cập nhật các biến theo dõi
                self.processed_count += 1
                self.last_processed_time = time.time()
                
                # Đánh dấu là đã hoàn thành
                self.queue.task_done()
                
                # Tự động lưu theo khoảng thời gian
                if self.processed_count % 10 == 0:  # Lưu sau mỗi 10 mục
                    self.save_queue()
                
            except Exception as e:
                self.logger.error(f"Lỗi trong vòng lặp xử lý: {str(e)}")
                time.sleep(1)  # Tránh loop quá nhanh khi có lỗi
        
        self.logger.info("Kết thúc vòng lặp xử lý hàng đợi")
    
    def _execute_strategy(self, strategy: Dict[str, Any], queue_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực thi chiến lược.
        
        Args:
            strategy: Thông tin chiến lược
            queue_item: Thông tin mục hàng đợi
            
        Returns:
            Kết quả thực thi
        """
        # Đây chỉ là hàm giả định, cần được triển khai cụ thể
        # Trong ứng dụng thực tế, hàm này sẽ gọi đến code thực thi chiến lược
        
        # Giả lập thực thi chiến lược
        time.sleep(1)  # Giả định mất 1 giây để thực thi
        
        # Tạo kết quả giả định
        result = {
            "executed_at": datetime.now().isoformat(),
            "strategy_id": strategy.get("id", "unknown"),
            "strategy_name": strategy.get("name", "unknown"),
            "profit": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "trades": 0
        }
        
        # Trong ứng dụng thực tế, kết quả sẽ được tính toán dựa trên kết quả thực thi chiến lược
        # Ví dụ: thực hiện backtest hoặc giao dịch thực
        
        return result