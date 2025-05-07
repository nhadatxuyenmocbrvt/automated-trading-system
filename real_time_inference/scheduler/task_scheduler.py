"""
Lập lịch tác vụ trong hệ thống.
File này quản lý việc lập lịch và thực thi các tác vụ tự động,
cung cấp chức năng lập lịch theo thời gian, tần suất và ưu tiên.
"""

import os
import time
import asyncio
import logging
import threading
import datetime
import heapq
from enum import Enum, auto
from typing import Dict, List, Callable, Any, Optional, Union, Tuple, Set
from pathlib import Path
from functools import wraps
from datetime import datetime, timedelta

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from real_time_inference.auto_restart.error_handler import handle_task_error

class TaskPriority(Enum):
    """Enum các mức độ ưu tiên cho tác vụ."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class TaskStatus(Enum):
    """Enum các trạng thái của tác vụ."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

class Task:
    """Đại diện cho một tác vụ trong hệ thống scheduler."""
    
    def __init__(
        self,
        task_id: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        schedule_time: Optional[datetime] = None,
        interval: Optional[timedelta] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
        timeout: Optional[int] = None,
        tags: List[str] = None
    ):
        """
        Khởi tạo một tác vụ.
        
        Args:
            task_id: ID duy nhất của tác vụ
            func: Hàm sẽ được thực thi
            args: Tham số vị trí cho hàm
            kwargs: Tham số từ khóa cho hàm
            priority: Độ ưu tiên của tác vụ
            schedule_time: Thời gian dự kiến thực thi
            interval: Khoảng thời gian lặp lại (None nếu không lặp lại)
            max_retries: Số lần thử lại tối đa khi gặp lỗi
            retry_delay: Thời gian chờ giữa các lần thử lại (giây)
            timeout: Thời gian tối đa cho phép thực thi (giây)
            tags: Danh sách các tag phân loại
        """
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.priority = priority
        self.schedule_time = schedule_time or datetime.now()
        self.interval = interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.tags = tags or []
        
        # Trạng thái
        self.status = TaskStatus.PENDING
        self.retry_count = 0
        self.last_run_time = None
        self.next_run_time = self.schedule_time
        self.result = None
        self.error = None
    
    def __lt__(self, other):
        """So sánh tác vụ để sắp xếp trong hàng đợi ưu tiên."""
        # So sánh trước tiên theo thời gian lịch
        if self.next_run_time != other.next_run_time:
            return self.next_run_time < other.next_run_time
        
        # Nếu thời gian lịch giống nhau, so sánh theo ưu tiên
        return self.priority.value > other.priority.value  # Ưu tiên cao hơn trước
    
    def reschedule(self):
        """Lên lịch lại tác vụ định kỳ."""
        if self.interval:
            self.next_run_time = datetime.now() + self.interval
            self.status = TaskStatus.PENDING
            self.retry_count = 0
            return True
        return False
    
    def to_dict(self) -> Dict:
        """
        Chuyển đổi tác vụ thành dictionary.
        
        Returns:
            Dictionary chứa thông tin tác vụ
        """
        return {
            'task_id': self.task_id,
            'function': self.func.__name__,
            'status': self.status.name,
            'priority': self.priority.name,
            'schedule_time': self.schedule_time.isoformat() if self.schedule_time else None,
            'next_run_time': self.next_run_time.isoformat() if self.next_run_time else None,
            'last_run_time': self.last_run_time.isoformat() if self.last_run_time else None,
            'interval': str(self.interval) if self.interval else None,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'tags': self.tags
        }

class TaskScheduler:
    """
    Lớp quản lý lập lịch và thực thi các tác vụ.
    Hỗ trợ lập lịch theo thời gian, ưu tiên và lặp lại.
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        check_interval: float = 1.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo scheduler.
        
        Args:
            max_workers: Số lượng tác vụ có thể chạy đồng thời tối đa
            check_interval: Khoảng thời gian kiểm tra các tác vụ (giây)
            logger: Logger tùy chỉnh
        """
        self.logger = logger or get_logger("task_scheduler")
        self.task_queue = []  # Priority queue
        self.running_tasks = set()  # Set of task_ids that are currently running
        self.tasks = {}  # Dict mapping task_id to Task
        self.max_workers = max_workers
        self.check_interval = check_interval
        
        # Locks
        self._queue_lock = threading.RLock()
        self._task_lock = threading.RLock()
        
        # Task Events
        self.task_events = {
            'on_start': [],
            'on_complete': [],
            'on_fail': [],
            'on_cancel': [],
            'on_reschedule': []
        }
        
        # Trạng thái scheduler
        self.is_running = False
        self._scheduler_thread = None
        self._stop_event = threading.Event()
        
        # Lấy cấu hình hệ thống
        self.config = get_system_config()
        
        self.logger.info("Đã khởi tạo Task Scheduler")
    
    def start(self):
        """Bắt đầu scheduler."""
        if self.is_running:
            self.logger.warning("Scheduler đã đang chạy")
            return
        
        self.is_running = True
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        self.logger.info("Đã bắt đầu Task Scheduler")
    
    def stop(self):
        """Dừng scheduler."""
        if not self.is_running:
            self.logger.warning("Scheduler không chạy")
            return
        
        self.is_running = False
        self._stop_event.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=30)
        
        self.logger.info("Đã dừng Task Scheduler")
    
    def _run_scheduler(self):
        """Vòng lặp chính của scheduler."""
        while not self._stop_event.is_set():
            self._process_due_tasks()
            time.sleep(self.check_interval)
    
    def _process_due_tasks(self):
        """Xử lý các tác vụ đến hạn thực thi."""
        now = datetime.now()
        tasks_to_run = []
        
        with self._queue_lock:
            # Lấy tất cả tác vụ đến hạn
            while self.task_queue and self.task_queue[0].next_run_time <= now:
                task = heapq.heappop(self.task_queue)
                if task.task_id not in self.running_tasks and task.status == TaskStatus.PENDING:
                    tasks_to_run.append(task)
                    self.running_tasks.add(task.task_id)
                else:
                    # Nếu tác vụ đang chạy, đưa lại vào hàng đợi
                    heapq.heappush(self.task_queue, task)
        
        # Thực thi các tác vụ đến hạn
        for task in tasks_to_run:
            self._execute_task(task)
    
    def _execute_task(self, task):
        """
        Thực thi một tác vụ.
        
        Args:
            task: Tác vụ cần thực thi
        """
        def task_wrapper():
            task.status = TaskStatus.RUNNING
            task.last_run_time = datetime.now()
            
            # Gọi các event handlers
            self._trigger_event('on_start', task)
            
            try:
                # Thực thi tác vụ với timeout nếu có
                if task.timeout:
                    # Tạo Future để thực thi với timeout
                    async def run_with_timeout():
                        try:
                            if asyncio.iscoroutinefunction(task.func):
                                # Nếu là coroutine
                                return await asyncio.wait_for(
                                    task.func(*task.args, **task.kwargs),
                                    timeout=task.timeout
                                )
                            else:
                                # Nếu là hàm thông thường
                                loop = asyncio.get_event_loop()
                                return await asyncio.wait_for(
                                    loop.run_in_executor(
                                        None, lambda: task.func(*task.args, **task.kwargs)
                                    ),
                                    timeout=task.timeout
                                )
                        except asyncio.TimeoutError:
                            raise TimeoutError(f"Task {task.task_id} timed out after {task.timeout} seconds")
                    
                    # Tạo vòng lặp event mới nếu cần
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Chạy coroutine với timeout
                    task.result = loop.run_until_complete(run_with_timeout())
                else:
                    # Thực thi không có timeout
                    if asyncio.iscoroutinefunction(task.func):
                        # Nếu là coroutine
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        task.result = loop.run_until_complete(task.func(*task.args, **task.kwargs))
                    else:
                        # Nếu là hàm thông thường
                        task.result = task.func(*task.args, **task.kwargs)
                
                task.status = TaskStatus.COMPLETED
                self._trigger_event('on_complete', task)
                self.logger.debug(f"Tác vụ {task.task_id} đã hoàn thành")
                
            except Exception as e:
                task.error = str(e)
                task.retry_count += 1
                self.logger.error(f"Lỗi khi thực thi tác vụ {task.task_id}: {str(e)}")
                
                # Xử lý lỗi với error handler
                try:
                    handle_task_error(task.task_id, e, task.to_dict())
                except Exception as err:
                    self.logger.error(f"Lỗi khi xử lý lỗi cho tác vụ {task.task_id}: {str(err)}")
                
                # Thử lại nếu chưa vượt quá số lần cho phép
                if task.retry_count < task.max_retries:
                    self.logger.info(f"Sẽ thử lại tác vụ {task.task_id} sau {task.retry_delay} giây (lần thứ {task.retry_count + 1})")
                    task.status = TaskStatus.PENDING
                    task.next_run_time = datetime.now() + timedelta(seconds=task.retry_delay)
                    
                    with self._queue_lock:
                        heapq.heappush(self.task_queue, task)
                else:
                    task.status = TaskStatus.FAILED
                    self._trigger_event('on_fail', task)
                    self.logger.warning(f"Tác vụ {task.task_id} đã thất bại sau {task.max_retries} lần thử")
            
            finally:
                # Lên lịch lại nếu là tác vụ định kỳ
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    if task.interval:
                        if task.reschedule():
                            self.logger.debug(f"Đã lên lịch lại tác vụ {task.task_id} vào {task.next_run_time}")
                            self._trigger_event('on_reschedule', task)
                            
                            with self._queue_lock:
                                heapq.heappush(self.task_queue, task)
                
                # Xóa khỏi danh sách các tác vụ đang chạy
                self.running_tasks.discard(task.task_id)
        
        # Chạy tác vụ trong thread riêng
        thread = threading.Thread(target=task_wrapper)
        thread.daemon = True
        thread.start()
    
    def schedule_task(self, task: Task) -> str:
        """
        Lên lịch cho một tác vụ.
        
        Args:
            task: Tác vụ cần lên lịch
            
        Returns:
            ID của tác vụ
        """
        with self._task_lock:
            # Kiểm tra xem ID đã tồn tại chưa
            if task.task_id in self.tasks:
                self.logger.warning(f"Tác vụ với ID {task.task_id} đã tồn tại, sẽ tạo ID mới")
                # Tạo ID mới bằng cách thêm timestamp
                timestamp = int(time.time())
                task.task_id = f"{task.task_id}_{timestamp}"
            
            self.tasks[task.task_id] = task
        
        with self._queue_lock:
            heapq.heappush(self.task_queue, task)
        
        self.logger.info(f"Đã lên lịch tác vụ {task.task_id} vào {task.next_run_time}")
        return task.task_id
    
    def schedule_function(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        task_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        schedule_time: Optional[datetime] = None,
        interval: Optional[timedelta] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
        timeout: Optional[int] = None,
        tags: List[str] = None
    ) -> str:
        """
        Lên lịch cho một hàm.
        
        Args:
            func: Hàm cần lên lịch
            args: Tham số vị trí cho hàm
            kwargs: Tham số từ khóa cho hàm
            task_id: ID của tác vụ (tùy chọn)
            priority: Độ ưu tiên của tác vụ
            schedule_time: Thời gian dự kiến thực thi
            interval: Khoảng thời gian lặp lại (None nếu không lặp lại)
            max_retries: Số lần thử lại tối đa khi gặp lỗi
            retry_delay: Thời gian chờ giữa các lần thử lại (giây)
            timeout: Thời gian tối đa cho phép thực thi (giây)
            tags: Danh sách các tag phân loại
            
        Returns:
            ID của tác vụ
        """
        # Tạo ID nếu không được cung cấp
        if task_id is None:
            timestamp = int(time.time())
            task_id = f"{func.__name__}_{timestamp}"
        
        # Tạo tác vụ
        task = Task(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            schedule_time=schedule_time,
            interval=interval,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            tags=tags
        )
        
        # Lên lịch tác vụ
        return self.schedule_task(task)
    
    def schedule_at(
        self,
        schedule_time: datetime,
        func: Callable,
        *args,
        **kwargs
    ) -> str:
        """
        Lên lịch một tác vụ tại thời điểm cụ thể.
        
        Args:
            schedule_time: Thời gian dự kiến thực thi
            func: Hàm cần lên lịch
            *args: Tham số vị trí cho hàm
            **kwargs: Tham số từ khóa cho hàm
            
        Returns:
            ID của tác vụ
        """
        task_kwargs = {}
        
        # Tách các tham số đặc biệt ra khỏi kwargs
        special_params = ['task_id', 'priority', 'max_retries', 'retry_delay', 'timeout', 'tags']
        for param in special_params:
            if param in kwargs:
                task_kwargs[param] = kwargs.pop(param)
        
        # Lên lịch tác vụ với thời gian cụ thể
        return self.schedule_function(
            func=func,
            args=args,
            kwargs=kwargs,
            schedule_time=schedule_time,
            **task_kwargs
        )
    
    def schedule_interval(
        self,
        interval: timedelta,
        func: Callable,
        *args,
        **kwargs
    ) -> str:
        """
        Lên lịch một tác vụ lặp lại theo khoảng thời gian.
        
        Args:
            interval: Khoảng thời gian lặp lại
            func: Hàm cần lên lịch
            *args: Tham số vị trí cho hàm
            **kwargs: Tham số từ khóa cho hàm
            
        Returns:
            ID của tác vụ
        """
        task_kwargs = {}
        
        # Tách các tham số đặc biệt ra khỏi kwargs
        special_params = ['task_id', 'priority', 'schedule_time', 'max_retries', 'retry_delay', 'timeout', 'tags']
        for param in special_params:
            if param in kwargs:
                task_kwargs[param] = kwargs.pop(param)
        
        # Lên lịch tác vụ với khoảng thời gian
        return self.schedule_function(
            func=func,
            args=args,
            kwargs=kwargs,
            interval=interval,
            **task_kwargs
        )
    
    def schedule_daily(
        self,
        hour: int,
        minute: int = 0,
        second: int = 0,
        func: Callable = None,
        *args,
        **kwargs
    ) -> str:
        """
        Lên lịch một tác vụ hàng ngày vào thời điểm cụ thể.
        
        Args:
            hour: Giờ (0-23)
            minute: Phút (0-59)
            second: Giây (0-59)
            func: Hàm cần lên lịch
            *args: Tham số vị trí cho hàm
            **kwargs: Tham số từ khóa cho hàm
            
        Returns:
            ID của tác vụ
        """
        # Tính toán thời điểm thực thi tiếp theo
        now = datetime.now()
        next_run = datetime(now.year, now.month, now.day, hour, minute, second)
        
        # Nếu thời gian đã qua trong ngày, chuyển sang ngày hôm sau
        if next_run <= now:
            next_run += timedelta(days=1)
        
        # Lên lịch tác vụ
        task_id = self.schedule_at(next_run, func, *args, **kwargs)
        
        # Thiết lập lặp lại hàng ngày
        with self._task_lock:
            if task_id in self.tasks:
                self.tasks[task_id].interval = timedelta(days=1)
        
        return task_id
    
    def schedule_weekly(
        self,
        day_of_week: int,  # 0: Thứ Hai, 6: Chủ Nhật
        hour: int,
        minute: int = 0,
        second: int = 0,
        func: Callable = None,
        *args,
        **kwargs
    ) -> str:
        """
        Lên lịch một tác vụ hàng tuần vào thời điểm cụ thể.
        
        Args:
            day_of_week: Ngày trong tuần (0: Thứ Hai, 6: Chủ Nhật)
            hour: Giờ (0-23)
            minute: Phút (0-59)
            second: Giây (0-59)
            func: Hàm cần lên lịch
            *args: Tham số vị trí cho hàm
            **kwargs: Tham số từ khóa cho hàm
            
        Returns:
            ID của tác vụ
        """
        # Tính toán thời điểm thực thi tiếp theo
        now = datetime.now()
        days_ahead = day_of_week - now.weekday()
        if days_ahead <= 0:  # Đã qua trong tuần này
            days_ahead += 7
        
        next_run = datetime(now.year, now.month, now.day, hour, minute, second) + timedelta(days=days_ahead)
        
        # Lên lịch tác vụ
        task_id = self.schedule_at(next_run, func, *args, **kwargs)
        
        # Thiết lập lặp lại hàng tuần
        with self._task_lock:
            if task_id in self.tasks:
                self.tasks[task_id].interval = timedelta(days=7)
        
        return task_id
    
    def schedule_monthly(
        self,
        day: int,
        hour: int,
        minute: int = 0,
        second: int = 0,
        func: Callable = None,
        *args,
        **kwargs
    ) -> str:
        """
        Lên lịch một tác vụ hàng tháng vào thời điểm cụ thể.
        
        Args:
            day: Ngày trong tháng (1-31)
            hour: Giờ (0-23)
            minute: Phút (0-59)
            second: Giây (0-59)
            func: Hàm cần lên lịch
            *args: Tham số vị trí cho hàm
            **kwargs: Tham số từ khóa cho hàm
            
        Returns:
            ID của tác vụ hoặc None nếu lịch không hợp lệ
        """
        if day < 1 or day > 31:
            self.logger.error(f"Ngày không hợp lệ: {day}. Phải nằm trong khoảng 1-31")
            return None
        
        # Tính toán thời điểm thực thi tiếp theo
        now = datetime.now()
        
        # Tìm ngày hợp lệ cho tháng tiếp theo
        next_month = now.month + 1 if now.month < 12 else 1
        next_year = now.year if now.month < 12 else now.year + 1
        
        # Điều chỉnh cho ngày cuối tháng (28/29/30/31)
        import calendar
        last_day = calendar.monthrange(next_year, next_month)[1]
        actual_day = min(day, last_day)
        
        next_run = datetime(next_year, next_month, actual_day, hour, minute, second)
        
        # Nếu thời gian tháng hiện tại vẫn còn phía trước
        if now.day < day:
            try:
                current_run = datetime(now.year, now.month, day, hour, minute, second)
                if current_run > now:
                    next_run = current_run
            except ValueError:
                # Ngày không hợp lệ cho tháng hiện tại
                pass
        
        # Lên lịch tác vụ
        task_id = self.schedule_at(next_run, func, *args, **kwargs)
        
        # Tạo hàm tính toán lần chạy tiếp theo cho tác vụ hàng tháng
        def monthly_reschedule(task):
            now = datetime.now()
            current_month = now.month
            current_year = now.year
            
            # Tìm tháng tiếp theo
            next_month = current_month + 1 if current_month < 12 else 1
            next_year = current_year if current_month < 12 else current_year + 1
            
            # Điều chỉnh cho ngày cuối tháng
            last_day = calendar.monthrange(next_year, next_month)[1]
            actual_day = min(day, last_day)
            
            next_run = datetime(next_year, next_month, actual_day, hour, minute, second)
            return next_run
        
        # Gán function reschedule tùy chỉnh cho tác vụ
        with self._task_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                
                # Ghi đè phương thức reschedule
                original_reschedule = task.reschedule
                
                def custom_reschedule():
                    task.next_run_time = monthly_reschedule(task)
                    task.status = TaskStatus.PENDING
                    task.retry_count = 0
                    return True
                
                task.reschedule = custom_reschedule
                task.interval = timedelta(days=28)  # Gần đúng, chỉ để đánh dấu là tác vụ định kỳ
        
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Hủy một tác vụ đã lên lịch.
        
        Args:
            task_id: ID của tác vụ cần hủy
            
        Returns:
            True nếu hủy thành công, False nếu không tìm thấy tác vụ
        """
        with self._task_lock:
            if task_id not in self.tasks:
                self.logger.warning(f"Không tìm thấy tác vụ có ID {task_id} để hủy")
                return False
            
            task = self.tasks[task_id]
            task.status = TaskStatus.CANCELLED
            
            # Gọi các event handlers
            self._trigger_event('on_cancel', task)
            
            # Xóa khỏi các cấu trúc dữ liệu (không xóa khỏi hàng đợi vì khó xóa từ heap)
            if task_id in self.running_tasks:
                self.running_tasks.discard(task_id)
            
            # Không xóa khỏi self.tasks để giữ lịch sử
            
            self.logger.info(f"Đã hủy tác vụ {task_id}")
            return True
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        """
        Lấy thông tin về một tác vụ.
        
        Args:
            task_id: ID của tác vụ
            
        Returns:
            Dictionary chứa thông tin tác vụ hoặc None nếu không tìm thấy
        """
        with self._task_lock:
            if task_id not in self.tasks:
                return None
            
            return self.tasks[task_id].to_dict()
    
    def get_all_tasks(self) -> List[Dict]:
        """
        Lấy thông tin về tất cả các tác vụ.
        
        Returns:
            Danh sách các dictionary chứa thông tin tác vụ
        """
        with self._task_lock:
            return [task.to_dict() for task in self.tasks.values()]
    
    def get_pending_tasks(self) -> List[Dict]:
        """
        Lấy danh sách các tác vụ đang chờ.
        
        Returns:
            Danh sách các dictionary chứa thông tin tác vụ đang chờ
        """
        with self._task_lock:
            return [task.to_dict() for task in self.tasks.values() 
                   if task.status == TaskStatus.PENDING]
    
    def get_running_tasks(self) -> List[Dict]:
        """
        Lấy danh sách các tác vụ đang chạy.
        
        Returns:
            Danh sách các dictionary chứa thông tin tác vụ đang chạy
        """
        with self._task_lock:
            return [task.to_dict() for task in self.tasks.values() 
                   if task.task_id in self.running_tasks]
    
    def get_tasks_by_tag(self, tag: str) -> List[Dict]:
        """
        Lấy danh sách các tác vụ theo tag.
        
        Args:
            tag: Tag cần tìm
            
        Returns:
            Danh sách các dictionary chứa thông tin tác vụ có tag tương ứng
        """
        with self._task_lock:
            return [task.to_dict() for task in self.tasks.values() 
                   if tag in task.tags]
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Dict]:
        """
        Lấy danh sách các tác vụ theo trạng thái.
        
        Args:
            status: Trạng thái cần tìm
            
        Returns:
            Danh sách các dictionary chứa thông tin tác vụ có trạng thái tương ứng
        """
        with self._task_lock:
            return [task.to_dict() for task in self.tasks.values() 
                   if task.status == status]
    
    def _trigger_event(self, event_type: str, task: Task):
        """
        Gọi các event handlers cho một sự kiện.
        
        Args:
            event_type: Loại sự kiện
            task: Tác vụ liên quan
        """
        if event_type in self.task_events:
            for handler in self.task_events[event_type]:
                try:
                    handler(task)
                except Exception as e:
                    self.logger.error(f"Lỗi khi gọi event handler {event_type}: {str(e)}")
    
    def register_event_handler(self, event_type: str, handler: Callable[[Task], None]):
        """
        Đăng ký một event handler.
        
        Args:
            event_type: Loại sự kiện ('on_start', 'on_complete', 'on_fail', 'on_cancel', 'on_reschedule')
            handler: Hàm xử lý sự kiện
        """
        if event_type not in self.task_events:
            self.logger.warning(f"Loại sự kiện không hỗ trợ: {event_type}")
            return
        
        self.task_events[event_type].append(handler)
        self.logger.debug(f"Đã đăng ký handler cho sự kiện {event_type}")
    
    def clear_completed_tasks(self, older_than: Optional[timedelta] = None):
        """
        Xóa các tác vụ đã hoàn thành khỏi bộ nhớ.
        
        Args:
            older_than: Chỉ xóa các tác vụ cũ hơn khoảng thời gian này
        """
        now = datetime.now()
        to_remove = []
        
        with self._task_lock:
            for task_id, task in self.tasks.items():
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    if task.last_run_time:
                        if older_than is None or now - task.last_run_time > older_than:
                            to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.tasks[task_id]
        
        self.logger.info(f"Đã xóa {len(to_remove)} tác vụ đã hoàn thành")

# Tạo singleton instance cho toàn bộ ứng dụng
_scheduler_instance = None

def get_scheduler(max_workers: int = 10, check_interval: float = 1.0, logger: Optional[logging.Logger] = None):
    """
    Lấy instance toàn cục của TaskScheduler (singleton pattern).
    
    Args:
        max_workers: Số lượng tác vụ có thể chạy đồng thời tối đa
        check_interval: Khoảng thời gian kiểm tra các tác vụ (giây)
        logger: Logger tùy chỉnh
        
    Returns:
        Instance của TaskScheduler
    """
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = TaskScheduler(
            max_workers=max_workers,
            check_interval=check_interval,
            logger=logger
        )
    return _scheduler_instance


def schedule(
    interval: Optional[Union[timedelta, tuple]] = None,
    at_time: Optional[Union[datetime, str]] = None,
    priority: TaskPriority = TaskPriority.MEDIUM,
    max_retries: int = 3,
    retry_delay: int = 5,
    timeout: Optional[int] = None,
    tags: List[str] = None
):
    """
    Decorator để lên lịch một hàm.
    
    Args:
        interval: Khoảng thời gian lặp lại hoặc tuple (ngày, giờ, phút, giây)
        at_time: Thời gian cụ thể để thực thi
        priority: Độ ưu tiên của tác vụ
        max_retries: Số lần thử lại tối đa khi gặp lỗi
        retry_delay: Thời gian chờ giữa các lần thử lại (giây)
        timeout: Thời gian tối đa cho phép thực thi (giây)
        tags: Danh sách các tag phân loại
        
    Returns:
        Decorator
    """
    def decorator(func):
        scheduler = get_scheduler()
        
        # Xử lý interval nếu là tuple
        td_interval = None
        if interval is not None:
            if isinstance(interval, tuple):
                days, hours, minutes, seconds = 0, 0, 0, 0
                if len(interval) > 0:
                    days = interval[0]
                if len(interval) > 1:
                    hours = interval[1]
                if len(interval) > 2:
                    minutes = interval[2]
                if len(interval) > 3:
                    seconds = interval[3]
                
                td_interval = timedelta(
                    days=days,
                    hours=hours,
                    minutes=minutes,
                    seconds=seconds
                )
            else:
                td_interval = interval
        
        # Xử lý at_time nếu là string
        dt_at_time = None
        if at_time is not None:
            if isinstance(at_time, str):
                # Format: "HH:MM:SS" hoặc "HH:MM"
                parts = at_time.split(":")
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
                second = int(parts[2]) if len(parts) > 2 else 0
                
                now = datetime.now()
                dt_at_time = datetime(now.year, now.month, now.day, hour, minute, second)
                
                # Nếu thời gian đã qua trong ngày, chuyển sang ngày hôm sau
                if dt_at_time <= now:
                    dt_at_time += timedelta(days=1)
            else:
                dt_at_time = at_time
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Lên lịch tác vụ
            if td_interval is not None:
                scheduler.schedule_function(
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    priority=priority,
                    interval=td_interval,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    timeout=timeout,
                    tags=tags
                )
            elif dt_at_time is not None:
                scheduler.schedule_at(
                    schedule_time=dt_at_time,
                    func=func,
                    *args,
                    priority=priority,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    timeout=timeout,
                    tags=tags,
                    **kwargs
                )
            else:
                # Lên lịch chạy ngay lập tức nếu không có interval và at_time
                scheduler.schedule_function(
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    priority=priority,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    timeout=timeout,
                    tags=tags
                )
            
            return func(*args, **kwargs)
        
        # Thêm tham chiếu đến hàm gốc
        wrapper.original_func = func
        return wrapper
    
    return decorator