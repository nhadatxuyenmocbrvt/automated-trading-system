"""
Quản lý hàng đợi chiến lược.
File này định nghĩa lớp QueueManager để quản lý các chiến lược giao dịch trong hàng đợi,
bao gồm thêm, xóa, sắp xếp và thực thi chiến lược dựa trên ưu tiên và hiệu suất.
"""

import os
import sys
import time
import logging
import threading
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from automation.strategy_queue.priority_system import PrioritySystem
from automation.metrics.performance_metrics import PerformanceTracker
from models.agents.base_agent import BaseAgent
from models.training_pipeline.trainer import Trainer
from automation.model_updater import ModelUpdater


class QueueManager:
    """
    Quản lý và điều phối hàng đợi các chiến lược giao dịch.
    Cung cấp các phương thức để thêm, xóa, sắp xếp và thực thi các chiến lược
    dựa trên ưu tiên, hiệu suất và điều kiện thị trường.
    """
    
    def __init__(
        self,
        max_active_strategies: int = 5,
        queue_check_interval: float = 60.0,
        performance_threshold: float = -0.05,
        auto_retraining: bool = True,
        max_retry_count: int = 3,
        strategies_dir: Optional[Union[str, Path]] = None,
        priority_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Khởi tạo trình quản lý hàng đợi chiến lược.
        
        Args:
            max_active_strategies: Số lượng chiến lược tối đa có thể hoạt động đồng thời
            queue_check_interval: Khoảng thời gian (giây) giữa các lần kiểm tra hàng đợi
            performance_threshold: Ngưỡng hiệu suất tối thiểu để duy trì chiến lược
            auto_retraining: Tự động tái huấn luyện chiến lược khi hiệu suất giảm
            max_retry_count: Số lần thử lại tối đa cho mỗi chiến lược khi thất bại
            strategies_dir: Thư mục chứa các chiến lược đã lưu
            priority_config: Cấu hình hệ thống ưu tiên
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("queue_manager")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Lưu trữ các tham số
        self.max_active_strategies = max_active_strategies
        self.queue_check_interval = queue_check_interval
        self.performance_threshold = performance_threshold
        self.auto_retraining = auto_retraining
        self.max_retry_count = max_retry_count
        self.kwargs = kwargs
        
        # Thiết lập thư mục chiến lược
        if strategies_dir is None:
            self.strategies_dir = Path(self.system_config.get("MODELS_DIR", "./models")) / "strategies"
        else:
            self.strategies_dir = Path(strategies_dir)
        
        self.strategies_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo các thành phần
        self.priority_system = PrioritySystem(config=priority_config, logger=self.logger)
        self.model_updater = ModelUpdater(logger=self.logger, **kwargs)
        self.performance_tracker = PerformanceTracker(logger=self.logger)
        
        # Các cấu trúc dữ liệu hàng đợi
        self.strategy_queue = []  # Danh sách các chiến lược trong hàng đợi
        self.active_strategies = {}  # Dict {strategy_id: strategy_info} cho các chiến lược đang chạy
        self.paused_strategies = {}  # Dict chiến lược đã tạm dừng
        self.failed_strategies = {}  # Dict chiến lược đã thất bại
        
        # Khóa đồng bộ
        self.queue_lock = threading.RLock()
        self.active_lock = threading.RLock()
        
        # Biến điều khiển
        self.running = False
        self.queue_thread = None
        self.execution_pool = ThreadPoolExecutor(max_workers=max_active_strategies)
        
        # Biến theo dõi
        self.last_queue_check = time.time()
        self.total_executions = 0
        self.successful_executions = 0
        
        self.logger.info(f"Đã khởi tạo QueueManager với {max_active_strategies} chiến lược tối đa")
    
    def start(self) -> bool:
        """
        Bắt đầu quản lý hàng đợi.
        
        Returns:
            bool: True nếu khởi động thành công, False nếu không
        """
        if self.running:
            self.logger.warning("QueueManager đã chạy")
            return False
        
        # Đặt trạng thái chạy
        self.running = True
        
        # Tải các chiến lược từ thư mục nếu có
        self._load_saved_strategies()
        
        # Khởi động thread kiểm tra hàng đợi
        self.queue_thread = threading.Thread(target=self._queue_check_loop, daemon=True)
        self.queue_thread.start()
        
        self.logger.info("QueueManager đã khởi động")
        return True
    
    def stop(self) -> bool:
        """
        Dừng quản lý hàng đợi.
        
        Returns:
            bool: True nếu dừng thành công, False nếu không
        """
        if not self.running:
            self.logger.warning("QueueManager không chạy")
            return False
        
        # Đặt trạng thái dừng
        self.running = False
        
        # Đợi thread kiểm tra hàng đợi kết thúc
        if self.queue_thread and self.queue_thread.is_alive():
            self.queue_thread.join(timeout=10.0)
        
        # Dừng các chiến lược đang chạy
        with self.active_lock:
            for strategy_id, strategy_info in list(self.active_strategies.items()):
                self._stop_strategy(strategy_id)
        
        # Lưu trạng thái hàng đợi
        self._save_queue_state()
        
        # Đóng thread pool
        self.execution_pool.shutdown(wait=True)
        
        self.logger.info("QueueManager đã dừng")
        return True
    
    def add_strategy(self, strategy_info: Dict[str, Any], priority: Optional[float] = None) -> str:
        """
        Thêm chiến lược vào hàng đợi.
        
        Args:
            strategy_info: Thông tin chiến lược
            priority: Ưu tiên ban đầu (None để tính tự động)
            
        Returns:
            str: ID chiến lược đã thêm
        """
        # Tạo ID duy nhất nếu chưa có
        if 'id' not in strategy_info:
            strategy_info['id'] = f"strategy_{int(time.time())}_{len(self.strategy_queue)}"
        
        strategy_id = strategy_info['id']
        
        # Đánh giá ưu tiên ban đầu nếu không được cung cấp
        if priority is None:
            priority = self.priority_system.calculate_initial_priority(strategy_info)
        
        # Tạo thông tin hàng đợi
        queue_item = {
            'strategy_info': strategy_info,
            'priority': priority,
            'added_time': datetime.now(),
            'last_execution': None,
            'executions': 0,
            'success_rate': 0.0,
            'avg_performance': 0.0,
            'retry_count': 0
        }
        
        # Thêm vào hàng đợi với khóa
        with self.queue_lock:
            self.strategy_queue.append(queue_item)
            # Sắp xếp lại hàng đợi
            self._sort_queue()
        
        self.logger.info(f"Đã thêm chiến lược '{strategy_id}' vào hàng đợi với ưu tiên {priority:.2f}")
        
        # Kiểm tra hàng đợi ngay lập tức nếu cần thiết
        if len(self.active_strategies) < self.max_active_strategies:
            self._check_queue()
        
        return strategy_id
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """
        Xóa chiến lược khỏi hàng đợi.
        
        Args:
            strategy_id: ID của chiến lược cần xóa
            
        Returns:
            bool: True nếu xóa thành công, False nếu không tìm thấy
        """
        # Kiểm tra xem chiến lược có đang hoạt động không
        with self.active_lock:
            if strategy_id in self.active_strategies:
                self._stop_strategy(strategy_id)
                self.logger.info(f"Đã dừng và xóa chiến lược đang chạy '{strategy_id}'")
                return True
        
        # Kiểm tra xem chiến lược có trong hàng đợi không
        with self.queue_lock:
            for i, item in enumerate(self.strategy_queue):
                if item['strategy_info']['id'] == strategy_id:
                    self.strategy_queue.pop(i)
                    self.logger.info(f"Đã xóa chiến lược '{strategy_id}' khỏi hàng đợi")
                    return True
        
        # Kiểm tra các chiến lược đã tạm dừng
        if strategy_id in self.paused_strategies:
            del self.paused_strategies[strategy_id]
            self.logger.info(f"Đã xóa chiến lược tạm dừng '{strategy_id}'")
            return True
        
        # Kiểm tra các chiến lược đã thất bại
        if strategy_id in self.failed_strategies:
            del self.failed_strategies[strategy_id]
            self.logger.info(f"Đã xóa chiến lược thất bại '{strategy_id}'")
            return True
        
        self.logger.warning(f"Không tìm thấy chiến lược '{strategy_id}' để xóa")
        return False
    
    def update_priority(self, strategy_id: str, new_priority: Optional[float] = None, 
                      delta: float = 0.0) -> bool:
        """
        Cập nhật ưu tiên cho chiến lược.
        
        Args:
            strategy_id: ID của chiến lược
            new_priority: Giá trị ưu tiên mới (None để điều chỉnh dựa trên delta)
            delta: Thay đổi ưu tiên (+/-)
            
        Returns:
            bool: True nếu cập nhật thành công, False nếu không
        """
        # Tìm kiếm trong hàng đợi
        found = False
        with self.queue_lock:
            for item in self.strategy_queue:
                if item['strategy_info']['id'] == strategy_id:
                    old_priority = item['priority']
                    if new_priority is not None:
                        item['priority'] = new_priority
                    else:
                        item['priority'] += delta
                    
                    self.logger.info(
                        f"Đã cập nhật ưu tiên chiến lược '{strategy_id}' từ {old_priority:.2f} thành {item['priority']:.2f}"
                    )
                    found = True
                    break
            
            # Sắp xếp lại hàng đợi nếu có thay đổi
            if found:
                self._sort_queue()
        
        return found
    
    def get_strategy_status(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy trạng thái của chiến lược.
        
        Args:
            strategy_id: ID của chiến lược
            
        Returns:
            Dict thông tin trạng thái hoặc None nếu không tìm thấy
        """
        # Kiểm tra trong các chiến lược đang chạy
        with self.active_lock:
            if strategy_id in self.active_strategies:
                result = self.active_strategies[strategy_id].copy()
                result['status'] = 'active'
                return result
        
        # Kiểm tra trong hàng đợi
        with self.queue_lock:
            for item in self.strategy_queue:
                if item['strategy_info']['id'] == strategy_id:
                    result = item.copy()
                    result['status'] = 'queued'
                    result['queue_position'] = self.strategy_queue.index(item) + 1
                    return result
        
        # Kiểm tra trong các chiến lược tạm dừng
        if strategy_id in self.paused_strategies:
            result = self.paused_strategies[strategy_id].copy()
            result['status'] = 'paused'
            return result
        
        # Kiểm tra trong các chiến lược thất bại
        if strategy_id in self.failed_strategies:
            result = self.failed_strategies[strategy_id].copy()
            result['status'] = 'failed'
            return result
        
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái tổng thể của hàng đợi.
        
        Returns:
            Dict chứa thông tin trạng thái hàng đợi
        """
        with self.queue_lock, self.active_lock:
            return {
                'active_strategies': len(self.active_strategies),
                'queued_strategies': len(self.strategy_queue),
                'paused_strategies': len(self.paused_strategies),
                'failed_strategies': len(self.failed_strategies),
                'max_active_strategies': self.max_active_strategies,
                'total_executions': self.total_executions,
                'successful_executions': self.successful_executions,
                'success_rate': (self.successful_executions / max(1, self.total_executions)) * 100,
                'next_in_queue': self._get_next_in_queue(count=3),
                'top_performing': self._get_top_performing(count=3),
                'last_queue_check': self.last_queue_check,
                'auto_retraining': self.auto_retraining
            }
    
    def pause_strategy(self, strategy_id: str) -> bool:
        """
        Tạm dừng chiến lược đang chạy.
        
        Args:
            strategy_id: ID chiến lược cần tạm dừng
            
        Returns:
            bool: True nếu tạm dừng thành công, False nếu không
        """
        with self.active_lock:
            if strategy_id in self.active_strategies:
                # Lưu thông tin chiến lược
                self.paused_strategies[strategy_id] = self.active_strategies[strategy_id].copy()
                self.paused_strategies[strategy_id]['paused_time'] = datetime.now()
                
                # Dừng chiến lược
                self._stop_strategy(strategy_id)
                
                self.logger.info(f"Đã tạm dừng chiến lược '{strategy_id}'")
                return True
        
        self.logger.warning(f"Không thể tạm dừng chiến lược '{strategy_id}': không tìm thấy hoặc không đang chạy")
        return False
    
    def resume_strategy(self, strategy_id: str, priority: Optional[float] = None) -> bool:
        """
        Tiếp tục chiến lược đã tạm dừng.
        
        Args:
            strategy_id: ID chiến lược cần tiếp tục
            priority: Ưu tiên mới (None để giữ nguyên)
            
        Returns:
            bool: True nếu tiếp tục thành công, False nếu không
        """
        if strategy_id in self.paused_strategies:
            # Lấy thông tin chiến lược đã tạm dừng
            strategy_info = self.paused_strategies[strategy_id]['strategy_info']
            
            # Xác định ưu tiên
            if priority is None:
                priority = self.paused_strategies[strategy_id].get('priority', 0.0)
            
            # Thêm lại vào hàng đợi
            self.add_strategy(strategy_info, priority)
            
            # Xóa khỏi danh sách tạm dừng
            del self.paused_strategies[strategy_id]
            
            self.logger.info(f"Đã tiếp tục chiến lược '{strategy_id}' với ưu tiên {priority:.2f}")
            return True
        
        self.logger.warning(f"Không thể tiếp tục chiến lược '{strategy_id}': không tìm thấy trong danh sách tạm dừng")
        return False
    
    def retry_failed_strategy(self, strategy_id: str, retraining: bool = True) -> bool:
        """
        Thử lại chiến lược đã thất bại.
        
        Args:
            strategy_id: ID chiến lược cần thử lại
            retraining: Tái huấn luyện trước khi thử lại
            
        Returns:
            bool: True nếu thử lại thành công, False nếu không
        """
        if strategy_id in self.failed_strategies:
            # Lấy thông tin chiến lược thất bại
            strategy_info = self.failed_strategies[strategy_id]['strategy_info']
            
            # Tái huấn luyện nếu cần
            if retraining:
                try:
                    strategy_info = self.model_updater.retrain_model(strategy_info)
                    self.logger.info(f"Đã tái huấn luyện chiến lược '{strategy_id}' trước khi thử lại")
                except Exception as e:
                    self.logger.error(f"Lỗi khi tái huấn luyện chiến lược '{strategy_id}': {str(e)}")
            
            # Tính ưu tiên mới
            priority = self.priority_system.calculate_retry_priority(self.failed_strategies[strategy_id])
            
            # Thêm lại vào hàng đợi
            self.add_strategy(strategy_info, priority)
            
            # Xóa khỏi danh sách thất bại
            del self.failed_strategies[strategy_id]
            
            self.logger.info(f"Đã thử lại chiến lược '{strategy_id}' với ưu tiên {priority:.2f}")
            return True
        
        self.logger.warning(f"Không thể thử lại chiến lược '{strategy_id}': không tìm thấy trong danh sách thất bại")
        return False
    
    def save_strategy(self, strategy_id: str) -> bool:
        """
        Lưu chiến lược vào thư mục.
        
        Args:
            strategy_id: ID chiến lược cần lưu
            
        Returns:
            bool: True nếu lưu thành công, False nếu không
        """
        strategy_status = self.get_strategy_status(strategy_id)
        if not strategy_status:
            self.logger.warning(f"Không thể lưu chiến lược '{strategy_id}': không tìm thấy")
            return False
        
        try:
            # Tạo đường dẫn file
            file_path = self.strategies_dir / f"{strategy_id}.json"
            
            # Lưu thông tin chiến lược
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(strategy_status, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Đã lưu chiến lược '{strategy_id}' tại {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu chiến lược '{strategy_id}': {str(e)}")
            return False
    
    def load_strategy(self, strategy_id: str) -> bool:
        """
        Tải chiến lược từ thư mục.
        
        Args:
            strategy_id: ID chiến lược cần tải
            
        Returns:
            bool: True nếu tải thành công, False nếu không
        """
        file_path = self.strategies_dir / f"{strategy_id}.json"
        
        if not file_path.exists():
            self.logger.warning(f"Không thể tải chiến lược '{strategy_id}': file không tồn tại")
            return False
        
        try:
            # Đọc file
            with open(file_path, 'r', encoding='utf-8') as f:
                strategy_data = json.load(f)
            
            # Thêm vào hàng đợi
            if 'strategy_info' in strategy_data:
                priority = strategy_data.get('priority', None)
                self.add_strategy(strategy_data['strategy_info'], priority)
                
                self.logger.info(f"Đã tải chiến lược '{strategy_id}' từ {file_path}")
                return True
            else:
                self.logger.warning(f"File chiến lược '{strategy_id}' không có thông tin cần thiết")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi khi tải chiến lược '{strategy_id}': {str(e)}")
            return False
    
    def _load_saved_strategies(self) -> int:
        """
        Tải tất cả chiến lược đã lưu từ thư mục.
        
        Returns:
            int: Số lượng chiến lược đã tải
        """
        loaded_count = 0
        for file_path in self.strategies_dir.glob("*.json"):
            try:
                strategy_id = file_path.stem
                if self.load_strategy(strategy_id):
                    loaded_count += 1
            except Exception as e:
                self.logger.error(f"Lỗi khi tải chiến lược từ {file_path}: {str(e)}")
        
        self.logger.info(f"Đã tải {loaded_count} chiến lược từ {self.strategies_dir}")
        return loaded_count
    
    def _save_queue_state(self) -> bool:
        """
        Lưu trạng thái hàng đợi.
        
        Returns:
            bool: True nếu lưu thành công, False nếu không
        """
        try:
            # Tạo đường dẫn file
            file_path = self.strategies_dir / "queue_state.json"
            
            # Tạo dữ liệu trạng thái
            with self.queue_lock, self.active_lock:
                state_data = {
                    'strategy_queue': self.strategy_queue,
                    'active_strategies': self.active_strategies,
                    'paused_strategies': self.paused_strategies,
                    'failed_strategies': self.failed_strategies,
                    'timestamp': datetime.now().isoformat(),
                    'total_executions': self.total_executions,
                    'successful_executions': self.successful_executions
                }
            
            # Lưu file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Đã lưu trạng thái hàng đợi tại {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu trạng thái hàng đợi: {str(e)}")
            return False
    
    def _load_queue_state(self) -> bool:
        """
        Tải trạng thái hàng đợi.
        
        Returns:
            bool: True nếu tải thành công, False nếu không
        """
        file_path = self.strategies_dir / "queue_state.json"
        
        if not file_path.exists():
            self.logger.info("Không tìm thấy file trạng thái hàng đợi để tải")
            return False
        
        try:
            # Đọc file
            with open(file_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # Cập nhật trạng thái
            with self.queue_lock, self.active_lock:
                self.strategy_queue = state_data.get('strategy_queue', [])
                self.paused_strategies = state_data.get('paused_strategies', {})
                self.failed_strategies = state_data.get('failed_strategies', {})
                self.total_executions = state_data.get('total_executions', 0)
                self.successful_executions = state_data.get('successful_executions', 0)
                
                # Không cập nhật active_strategies vì đang không chạy
            
            self.logger.info(f"Đã tải trạng thái hàng đợi từ {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tải trạng thái hàng đợi: {str(e)}")
            return False
    
    def _queue_check_loop(self) -> None:
        """
        Vòng lặp kiểm tra hàng đợi liên tục.
        """
        self.logger.info("Bắt đầu vòng lặp kiểm tra hàng đợi")
        
        while self.running:
            try:
                # Kiểm tra hàng đợi
                self._check_queue()
                
                # Cập nhật ưu tiên của tất cả chiến lược trong hàng đợi
                self._update_all_priorities()
                
                # Lưu trạng thái định kỳ
                self._periodic_save()
                
                # Đợi đến lần kiểm tra tiếp theo
                time.sleep(self.queue_check_interval)
                
            except Exception as e:
                self.logger.error(f"Lỗi trong vòng lặp kiểm tra hàng đợi: {str(e)}")
                time.sleep(5.0)  # Chờ một lát trước khi thử lại
        
        self.logger.info("Kết thúc vòng lặp kiểm tra hàng đợi")
    
    def _check_queue(self) -> None:
        """
        Kiểm tra hàng đợi và thực thi các chiến lược tiếp theo nếu có chỗ.
        """
        self.last_queue_check = time.time()
        
        # Kiểm tra hiệu suất của các chiến lược đang chạy
        self._check_active_strategies()
        
        # Khởi chạy các chiến lược tiếp theo nếu có chỗ
        with self.active_lock:
            available_slots = self.max_active_strategies - len(self.active_strategies)
        
        if available_slots <= 0:
            return
        
        # Lấy chiến lược tiếp theo từ hàng đợi
        next_strategies = self._get_next_strategies(available_slots)
        
        # Khởi chạy các chiến lược
        for strategy in next_strategies:
            strategy_id = strategy['strategy_info']['id']
            self._execute_strategy(strategy)
    
    def _update_all_priorities(self) -> None:
        """
        Cập nhật ưu tiên cho tất cả chiến lược trong hàng đợi.
        """
        with self.queue_lock:
            # Cập nhật từng chiến lược
            for item in self.strategy_queue:
                # Tính toán ưu tiên mới dựa trên hiệu suất, thời gian chờ, v.v.
                new_priority = self.priority_system.recalculate_priority(item)
                item['priority'] = new_priority
            
            # Sắp xếp lại hàng đợi
            self._sort_queue()
    
    def _sort_queue(self) -> None:
        """
        Sắp xếp hàng đợi theo ưu tiên.
        """
        # Đã được bảo vệ bởi queue_lock
        self.strategy_queue.sort(key=lambda x: x['priority'], reverse=True)
    
    def _get_next_strategies(self, count: int) -> List[Dict[str, Any]]:
        """
        Lấy các chiến lược tiếp theo từ hàng đợi.
        
        Args:
            count: Số lượng chiến lược cần lấy
            
        Returns:
            List các chiến lược tiếp theo
        """
        result = []
        
        with self.queue_lock:
            # Sắp xếp để đảm bảo lấy đúng thứ tự ưu tiên
            self._sort_queue()
            
            # Lấy các chiến lược đầu hàng đợi
            for _ in range(min(count, len(self.strategy_queue))):
                if self.strategy_queue:
                    result.append(self.strategy_queue.pop(0))
        
        return result
    
    def _execute_strategy(self, strategy_item: Dict[str, Any]) -> None:
        """
        Thực thi chiến lược.
        
        Args:
            strategy_item: Thông tin chiến lược cần thực thi
        """
        strategy_info = strategy_item['strategy_info']
        strategy_id = strategy_info['id']
        
        self.logger.info(f"Bắt đầu thực thi chiến lược '{strategy_id}' với ưu tiên {strategy_item['priority']:.2f}")
        
        # Thêm vào danh sách đang chạy
        with self.active_lock:
            self.active_strategies[strategy_id] = strategy_item.copy()
            self.active_strategies[strategy_id]['start_time'] = datetime.now()
            self.active_strategies[strategy_id]['status'] = 'running'
        
        # Tăng số lần thực thi
        self.total_executions += 1
        strategy_item['executions'] += 1
        strategy_item['last_execution'] = datetime.now()
        
        # Thực thi chiến lược trong thread riêng biệt
        self.execution_pool.submit(self._strategy_execution_task, strategy_id, strategy_info)
    
    def _strategy_execution_task(self, strategy_id: str, strategy_info: Dict[str, Any]) -> None:
        """
        Task thực thi chiến lược trong thread riêng biệt.
        
        Args:
            strategy_id: ID chiến lược
            strategy_info: Thông tin chiến lược
        """
        try:
            # Gọi phương thức thực thi từ thông tin chiến lược
            executor_class = strategy_info.get('executor_class')
            executor_params = strategy_info.get('executor_params', {})
            
            # Tạo executor nếu có
            if executor_class and hasattr(sys.modules[__name__], executor_class):
                executor = getattr(sys.modules[__name__], executor_class)(**executor_params)
                result = executor.execute(strategy_info)
            else:
                # Mặc định sử dụng mô hình cho forward inference
                agent_path = strategy_info.get('agent_path')
                agent_type = strategy_info.get('agent_type', 'dqn')
                
                # Tải agent từ đường dẫn
                # TODO: Implement appropriate loading mechanism based on agent_type
                agent = BaseAgent.load_from_path(agent_path, agent_type)
                
                # Chạy forward inference
                result = agent.run_inference(strategy_info.get('inference_params', {}))
            
            # Cập nhật kết quả thực thi
            self._handle_strategy_result(strategy_id, result, success=True)
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thực thi chiến lược '{strategy_id}': {str(e)}")
            
            # Đánh dấu thất bại
            self._handle_strategy_result(strategy_id, {"error": str(e)}, success=False)
    
    def _handle_strategy_result(self, strategy_id: str, result: Dict[str, Any], success: bool) -> None:
        """
        Xử lý kết quả thực thi chiến lược.
        
        Args:
            strategy_id: ID chiến lược
            result: Kết quả thực thi
            success: Thành công hay thất bại
        """
        with self.active_lock:
            if strategy_id not in self.active_strategies:
                self.logger.warning(f"Không tìm thấy chiến lược '{strategy_id}' trong danh sách đang chạy để cập nhật kết quả")
                return
            
            # Lấy thông tin chiến lược
            strategy_item = self.active_strategies[strategy_id]
            strategy_item['end_time'] = datetime.now()
            strategy_item['execution_time'] = (strategy_item['end_time'] - strategy_item['start_time']).total_seconds()
            strategy_item['result'] = result
            strategy_item['success'] = success
            
            # Cập nhật số liệu thành công
            if success:
                self.successful_executions += 1
                
                # Cập nhật hiệu suất
                performance = result.get('performance', 0.0)
                strategy_item['last_performance'] = performance
                
                # Cập nhật hiệu suất trung bình
                prev_avg = strategy_item.get('avg_performance', 0.0)
                n = strategy_item['executions']
                strategy_item['avg_performance'] = (prev_avg * (n-1) + performance) / n
                
                # Cập nhật tỷ lệ thành công
                strategy_item['success_rate'] = (
                    strategy_item.get('success_rate', 0) * (n-1) + 1
                ) / n
                
                # Lưu kết quả vào performance tracker
                self.performance_tracker.add_performance_record(
                    strategy_id=strategy_id,
                    timestamp=strategy_item['end_time'],
                    execution_time=strategy_item['execution_time'],
                    performance=performance,
                    result=result
                )
                
                # Xóa khỏi danh sách đang chạy
                strategy_copy = strategy_item.copy()
                del self.active_strategies[strategy_id]
                
                # Thêm lại vào hàng đợi với ưu tiên mới
                with self.queue_lock:
                    # Tính ưu tiên mới dựa trên kết quả
                    new_priority = self.priority_system.calculate_priority_after_execution(
                        strategy_copy, performance, success=True
                    )
                    
                    # Thêm lại vào hàng đợi
                    self.strategy_queue.append({
                        'strategy_info': strategy_copy['strategy_info'],
                        'priority': new_priority,
                        'added_time': datetime.now(),
                        'last_execution': strategy_copy['end_time'],
                        'executions': strategy_copy['executions'],
                        'success_rate': strategy_copy['success_rate'],
                        'avg_performance': strategy_copy['avg_performance'],
                        'retry_count': 0  # Reset retry count sau thành công
                    })
                    
                    # Sắp xếp lại hàng đợi
                    self._sort_queue()
                
                self.logger.info(
                    f"Chiến lược '{strategy_id}' thực thi thành công, performance: {performance:.4f}, "
                    f"đã thêm lại vào hàng đợi với ưu tiên {new_priority:.2f}"
                )
                
                # Tái huấn luyện nếu hiệu suất không tốt
                if self.auto_retraining and performance < self.performance_threshold:
                    self.logger.info(
                        f"Hiệu suất chiến lược '{strategy_id}' ({performance:.4f}) dưới ngưỡng "
                        f"({self.performance_threshold}), sẽ tái huấn luyện"
                    )
                    self._schedule_retraining(strategy_id, strategy_copy['strategy_info'])
            
            else:
                # Xử lý thất bại
                retry_count = strategy_item.get('retry_count', 0)
                
                if retry_count < self.max_retry_count:
                    # Tăng số lần thử và thêm lại vào hàng đợi với ưu tiên thấp hơn
                    strategy_item['retry_count'] = retry_count + 1
                    
                    # Tính ưu tiên mới
                    new_priority = self.priority_system.calculate_priority_after_execution(
                        strategy_item, 0.0, success=False
                    )
                    
                    # Thêm lại vào hàng đợi
                    with self.queue_lock:
                        self.strategy_queue.append({
                            'strategy_info': strategy_item['strategy_info'],
                            'priority': new_priority,
                            'added_time': datetime.now(),
                            'last_execution': strategy_item['end_time'],
                            'executions': strategy_item['executions'],
                            'success_rate': strategy_item.get('success_rate', 0),
                            'avg_performance': strategy_item.get('avg_performance', 0),
                            'retry_count': retry_count + 1,
                            'last_error': result.get('error', 'Unknown error')
                        })
                        
                        # Sắp xếp lại hàng đợi
                        self._sort_queue()
                    
                    self.logger.warning(
                        f"Chiến lược '{strategy_id}' thất bại, thử lại lần {retry_count + 1}/{self.max_retry_count}, "
                        f"đã thêm lại vào hàng đợi với ưu tiên {new_priority:.2f}"
                    )
                
                else:
                    # Đã đạt số lần thử tối đa, chuyển sang danh sách thất bại
                    self.failed_strategies[strategy_id] = strategy_item.copy()
                    self.failed_strategies[strategy_id]['failure_time'] = datetime.now()
                    
                    self.logger.error(
                        f"Chiến lược '{strategy_id}' đã thất bại {retry_count + 1} lần, "
                        f"đã chuyển sang danh sách thất bại"
                    )
                
                # Xóa khỏi danh sách đang chạy
                del self.active_strategies[strategy_id]
    
    def _schedule_retraining(self, strategy_id: str, strategy_info: Dict[str, Any]) -> None:
        """
        Lập lịch tái huấn luyện chiến lược.
        
        Args:
            strategy_id: ID chiến lược
            strategy_info: Thông tin chiến lược
        """
        try:
            # Thêm vào hàng đợi tái huấn luyện của model_updater
            self.model_updater.schedule_retraining(strategy_id, strategy_info)
            self.logger.info(f"Đã lập lịch tái huấn luyện cho chiến lược '{strategy_id}'")
        except Exception as e:
            self.logger.error(f"Lỗi khi lập lịch tái huấn luyện cho chiến lược '{strategy_id}': {str(e)}")
    
    def _check_active_strategies(self) -> None:
        """
        Kiểm tra trạng thái và hiệu suất của các chiến lược đang chạy.
        """
        with self.active_lock:
            # Kiểm tra từng chiến lược
            for strategy_id, strategy_info in list(self.active_strategies.items()):
                # Kiểm tra xem chiến lược có bị treo không
                start_time = strategy_info.get('start_time')
                if start_time:
                    running_time = (datetime.now() - start_time).total_seconds()
                    
                    # Nếu chạy quá lâu, có thể bị treo
                    max_execution_time = strategy_info.get('max_execution_time', 3600.0)  # Mặc định 1 giờ
                    
                    if running_time > max_execution_time:
                        self.logger.warning(
                            f"Chiến lược '{strategy_id}' đã chạy quá lâu ({running_time:.1f}s > {max_execution_time:.1f}s), "
                            f"có thể bị treo, sẽ dừng và thử lại"
                        )
                        
                        # Đánh dấu thất bại và thêm lại vào hàng đợi
                        self._handle_strategy_result(
                            strategy_id,
                            {"error": f"Execution timeout after {running_time:.1f}s"},
                            success=False
                        )
    
    def _stop_strategy(self, strategy_id: str) -> bool:
        """
        Dừng chiến lược đang chạy.
        
        Args:
            strategy_id: ID chiến lược cần dừng
            
        Returns:
            bool: True nếu dừng thành công, False nếu không
        """
        # Đã được bảo vệ bởi active_lock
        if strategy_id not in self.active_strategies:
            return False
        
        # TODO: Implement actual stopping mechanism
        # This is just a placeholder, actual implementation depends on
        # how strategies are executed and if they can be interrupted
        
        # Xóa khỏi danh sách đang chạy
        del self.active_strategies[strategy_id]
        
        return True
    
    def _get_next_in_queue(self, count: int = 3) -> List[Dict[str, Any]]:
        """
        Lấy thông tin về các chiến lược tiếp theo trong hàng đợi.
        
        Args:
            count: Số lượng chiến lược cần lấy
            
        Returns:
            List các thông tin chiến lược
        """
        result = []
        
        with self.queue_lock:
            # Sao chép để tránh thay đổi hàng đợi thật
            sorted_queue = sorted(self.strategy_queue, key=lambda x: x['priority'], reverse=True)
            
            # Lấy các chiến lược đầu hàng đợi
            for i, item in enumerate(sorted_queue[:count]):
                result.append({
                    'id': item['strategy_info']['id'],
                    'name': item['strategy_info'].get('name', f"Strategy {i+1}"),
                    'priority': item['priority'],
                    'added_time': item['added_time'].isoformat() if isinstance(item['added_time'], datetime) else item['added_time'],
                    'executions': item['executions'],
                    'avg_performance': item.get('avg_performance', 0.0)
                })
        
        return result
    
    def _get_top_performing(self, count: int = 3) -> List[Dict[str, Any]]:
        """
        Lấy thông tin về các chiến lược có hiệu suất tốt nhất.
        
        Args:
            count: Số lượng chiến lược cần lấy
            
        Returns:
            List các thông tin chiến lược
        """
        # Kết hợp tất cả chiến lược từ các danh sách
        all_strategies = []
        
        with self.queue_lock, self.active_lock:
            # Từ hàng đợi
            for item in self.strategy_queue:
                if item.get('executions', 0) > 0:  # Chỉ lấy các chiến lược đã chạy ít nhất một lần
                    all_strategies.append({
                        'id': item['strategy_info']['id'],
                        'name': item['strategy_info'].get('name', item['strategy_info']['id']),
                        'avg_performance': item.get('avg_performance', 0.0),
                        'executions': item.get('executions', 0),
                        'success_rate': item.get('success_rate', 0.0),
                        'status': 'queued'
                    })
            
            # Từ đang chạy
            for strategy_id, item in self.active_strategies.items():
                if item.get('executions', 0) > 0:
                    all_strategies.append({
                        'id': strategy_id,
                        'name': item['strategy_info'].get('name', strategy_id),
                        'avg_performance': item.get('avg_performance', 0.0),
                        'executions': item.get('executions', 0),
                        'success_rate': item.get('success_rate', 0.0),
                        'status': 'active'
                    })
            
            # Từ tạm dừng
            for strategy_id, item in self.paused_strategies.items():
                if item.get('executions', 0) > 0:
                    all_strategies.append({
                        'id': strategy_id,
                        'name': item['strategy_info'].get('name', strategy_id),
                        'avg_performance': item.get('avg_performance', 0.0),
                        'executions': item.get('executions', 0),
                        'success_rate': item.get('success_rate', 0.0),
                        'status': 'paused'
                    })
        
        # Sắp xếp theo hiệu suất giảm dần
        all_strategies.sort(key=lambda x: x['avg_performance'], reverse=True)
        
        # Trả về top N
        return all_strategies[:count]
    
    def _periodic_save(self) -> None:
        """
        Lưu trạng thái định kỳ.
        """
        # Lưu trạng thái hàng đợi mỗi giờ hoặc khi có thay đổi lớn
        current_time = time.time()
        last_save_time = getattr(self, '_last_save_time', 0)
        
        if current_time - last_save_time > 3600:  # 1 giờ
            self._save_queue_state()
            self._last_save_time = current_time