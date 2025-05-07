"""
Quản lý công việc định kỳ theo cron.
File này cung cấp các lớp và phương thức để định nghĩa, lên lịch và quản lý
các công việc định kỳ theo cú pháp cron, hỗ trợ đa dạng loại công việc tự động.
"""

import os
import re
import time
import asyncio
import logging
import threading
from enum import Enum, auto
from typing import Dict, List, Any, Callable, Optional, Union, Set, Tuple
from datetime import datetime, timedelta
from croniter import croniter
from pathlib import Path

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from real_time_inference.auto_restart.error_handler import handle_task_error
from real_time_inference.scheduler.task_scheduler import (
    Task, TaskPriority, TaskStatus, TaskScheduler, get_scheduler
)

class CronJob:
    """Định nghĩa một công việc định kỳ theo cú pháp cron."""
    
    def __init__(
        self,
        job_id: str,
        cron_expression: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        description: str = "",
        enabled: bool = True,
        max_instances: int = 1,
        priority: TaskPriority = TaskPriority.MEDIUM,
        max_retries: int = 3,
        retry_delay: int = 5,
        timeout: Optional[int] = None,
        tags: List[str] = None
    ):
        """
        Khởi tạo một công việc cron.
        
        Args:
            job_id: ID duy nhất của công việc
            cron_expression: Biểu thức cron (ví dụ: "0 * * * *" cho mỗi giờ)
            func: Hàm sẽ được thực thi
            args: Tham số vị trí cho hàm
            kwargs: Tham số từ khóa cho hàm
            description: Mô tả công việc
            enabled: Trạng thái kích hoạt ban đầu
            max_instances: Số lượng instances tối đa có thể chạy đồng thời
            priority: Độ ưu tiên của tác vụ
            max_retries: Số lần thử lại tối đa khi gặp lỗi
            retry_delay: Thời gian chờ giữa các lần thử lại (giây)
            timeout: Thời gian tối đa cho phép thực thi (giây)
            tags: Danh sách các tag phân loại
        """
        self.job_id = job_id
        self.cron_expression = cron_expression
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.description = description
        self.enabled = enabled
        self.max_instances = max_instances
        self.priority = priority
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.tags = tags or []
        
        # Validate cron expression
        try:
            croniter(cron_expression)
        except ValueError as e:
            raise ValueError(f"Biểu thức cron không hợp lệ '{cron_expression}': {str(e)}")
        
        # Trạng thái
        self.running_instances = 0
        self.last_run_time = None
        self.next_run_time = None
        self.last_status = None
        self.error = None
        
        # Tính toán thời gian chạy tiếp theo
        self._update_next_run_time()
    
    def _update_next_run_time(self):
        """Cập nhật thời gian chạy tiếp theo dựa trên biểu thức cron."""
        base_time = datetime.now()
        cron = croniter(self.cron_expression, base_time)
        self.next_run_time = cron.get_next(datetime)
    
    def to_dict(self) -> Dict:
        """
        Chuyển đổi công việc thành dictionary.
        
        Returns:
            Dictionary chứa thông tin công việc
        """
        return {
            'job_id': self.job_id,
            'cron_expression': self.cron_expression,
            'function': self.func.__name__,
            'description': self.description,
            'enabled': self.enabled,
            'running_instances': self.running_instances,
            'max_instances': self.max_instances,
            'priority': self.priority.name,
            'last_run_time': self.last_run_time.isoformat() if self.last_run_time else None,
            'next_run_time': self.next_run_time.isoformat() if self.next_run_time else None,
            'last_status': self.last_status,
            'error': self.error,
            'tags': self.tags
        }

class CronJobManager:
    """
    Lớp quản lý các công việc định kỳ theo cron.
    Điều phối việc lên lịch và theo dõi các công việc cron.
    """
    
    def __init__(
        self,
        task_scheduler: Optional[TaskScheduler] = None,
        check_interval: float = 60.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo CronJobManager.
        
        Args:
            task_scheduler: Task scheduler để lên lịch tác vụ (tùy chọn)
            check_interval: Khoảng thời gian kiểm tra các công việc (giây)
            logger: Logger tùy chỉnh
        """
        self.logger = logger or get_logger("cron_jobs")
        
        # Sử dụng task scheduler được cung cấp hoặc lấy instance mặc định
        self.task_scheduler = task_scheduler or get_scheduler()
        self.check_interval = check_interval
        
        # Lưu trữ các công việc
        self.jobs = {}  # Dict mapping job_id to CronJob
        
        # Locks
        self._jobs_lock = threading.RLock()
        
        # Cờ trạng thái
        self.is_running = False
        self._stop_event = threading.Event()
        self._scheduler_thread = None
        
        # Lấy cấu hình hệ thống
        self.config = get_system_config()
        
        # Task IDs đang chạy
        self.running_task_ids = {}  # Dict mapping job_id to set of task_ids
        
        self.logger.info("Đã khởi tạo Cron Job Manager")
    
    def start(self):
        """Bắt đầu manager."""
        if self.is_running:
            self.logger.warning("Cron Job Manager đã đang chạy")
            return
        
        # Đảm bảo task scheduler đang chạy
        if not self.task_scheduler.is_running:
            self.task_scheduler.start()
        
        self.is_running = True
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        self.logger.info("Đã bắt đầu Cron Job Manager")
    
    def stop(self):
        """Dừng manager."""
        if not self.is_running:
            self.logger.warning("Cron Job Manager không chạy")
            return
        
        self.is_running = False
        self._stop_event.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=30)
        
        self.logger.info("Đã dừng Cron Job Manager")
    
    def _run_scheduler(self):
        """Vòng lặp chính của manager."""
        while not self._stop_event.is_set():
            self._check_and_schedule_jobs()
            time.sleep(self.check_interval)
    
    def _check_and_schedule_jobs(self):
        """Kiểm tra và lên lịch các công việc đến hạn."""
        now = datetime.now()
        
        with self._jobs_lock:
            for job_id, job in list(self.jobs.items()):
                if not job.enabled:
                    continue
                
                # Kiểm tra xem công việc có đến hạn chưa
                if job.next_run_time and job.next_run_time <= now:
                    # Kiểm tra số lượng instances đang chạy
                    if job.running_instances < job.max_instances:
                        self._schedule_job(job)
                    else:
                        self.logger.warning(
                            f"Công việc {job_id} đã đạt số lượng instances tối đa ({job.max_instances}), "
                            f"bỏ qua lần chạy này"
                        )
                    
                    # Cập nhật thời gian chạy tiếp theo
                    job._update_next_run_time()
    
    def _schedule_job(self, job: CronJob):
        """
        Lên lịch một công việc cron.
        
        Args:
            job: Công việc cần lên lịch
        """
        # Tạo wrapper cho công việc
        def job_wrapper():
            # Cập nhật trạng thái
            with self._jobs_lock:
                job.running_instances += 1
                job.last_run_time = datetime.now()
                job.last_status = "RUNNING"
            
            try:
                # Thực thi công việc
                result = job.func(*job.args, **job.kwargs)
                
                # Cập nhật trạng thái thành công
                with self._jobs_lock:
                    job.last_status = "COMPLETED"
                    job.error = None
                
                return result
                
            except Exception as e:
                # Xử lý lỗi
                error_msg = str(e)
                self.logger.error(f"Lỗi khi thực thi công việc {job.job_id}: {error_msg}")
                
                # Cập nhật trạng thái lỗi
                with self._jobs_lock:
                    job.last_status = "FAILED"
                    job.error = error_msg
                
                # Báo cáo lỗi với error handler
                try:
                    handle_task_error(job.job_id, e, job.to_dict())
                except Exception as err:
                    self.logger.error(f"Lỗi khi xử lý lỗi cho công việc {job.job_id}: {str(err)}")
                
                # Re-raise để task scheduler xử lý retry
                raise
                
            finally:
                # Giảm số lượng instances đang chạy
                with self._jobs_lock:
                    job.running_instances = max(0, job.running_instances - 1)
                
                # Xóa task_id khỏi danh sách đang chạy
                if job.job_id in self.running_task_ids:
                    for task_id in list(self.running_task_ids.get(job.job_id, set())):
                        task = self.task_scheduler.get_task(task_id)
                        if task and task.get('status', '') in ['COMPLETED', 'FAILED']:
                            self.running_task_ids[job.job_id].discard(task_id)
        
        # Lên lịch công việc trên task scheduler
        task_id = self.task_scheduler.schedule_function(
            func=job_wrapper,
            priority=job.priority,
            max_retries=job.max_retries,
            retry_delay=job.retry_delay,
            timeout=job.timeout,
            tags=job.tags + ['cron_job', f'job_id:{job.job_id}']
        )
        
        # Lưu task_id vào danh sách đang chạy
        with self._jobs_lock:
            if job.job_id not in self.running_task_ids:
                self.running_task_ids[job.job_id] = set()
            self.running_task_ids[job.job_id].add(task_id)
        
        self.logger.info(f"Đã lên lịch công việc {job.job_id} (task_id: {task_id})")
    
    def add_job(self, job: CronJob) -> str:
        """
        Thêm một công việc cron vào manager.
        
        Args:
            job: Công việc cron cần thêm
            
        Returns:
            ID của công việc
        """
        with self._jobs_lock:
            # Kiểm tra xem ID đã tồn tại chưa
            if job.job_id in self.jobs:
                self.logger.warning(f"Công việc với ID {job.job_id} đã tồn tại, sẽ tạo ID mới")
                # Tạo ID mới bằng cách thêm timestamp
                timestamp = int(time.time())
                job.job_id = f"{job.job_id}_{timestamp}"
            
            # Thêm công việc vào danh sách
            self.jobs[job.job_id] = job
            self.running_task_ids[job.job_id] = set()
        
        self.logger.info(f"Đã thêm công việc cron {job.job_id} ({job.cron_expression}): {job.description}")
        return job.job_id
    
    def schedule_job(
        self,
        cron_expression: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        job_id: Optional[str] = None,
        description: str = "",
        enabled: bool = True,
        max_instances: int = 1,
        priority: TaskPriority = TaskPriority.MEDIUM,
        max_retries: int = 3,
        retry_delay: int = 5,
        timeout: Optional[int] = None,
        tags: List[str] = None
    ) -> str:
        """
        Lên lịch một công việc cron.
        
        Args:
            cron_expression: Biểu thức cron
            func: Hàm cần lên lịch
            args: Tham số vị trí cho hàm
            kwargs: Tham số từ khóa cho hàm
            job_id: ID của công việc (tùy chọn)
            description: Mô tả công việc
            enabled: Trạng thái kích hoạt ban đầu
            max_instances: Số lượng instances tối đa có thể chạy đồng thời
            priority: Độ ưu tiên của tác vụ
            max_retries: Số lần thử lại tối đa khi gặp lỗi
            retry_delay: Thời gian chờ giữa các lần thử lại (giây)
            timeout: Thời gian tối đa cho phép thực thi (giây)
            tags: Danh sách các tag phân loại
            
        Returns:
            ID của công việc
        """
        # Tạo ID nếu không được cung cấp
        if job_id is None:
            timestamp = int(time.time())
            job_id = f"{func.__name__}_{timestamp}"
        
        # Tạo công việc cron
        job = CronJob(
            job_id=job_id,
            cron_expression=cron_expression,
            func=func,
            args=args,
            kwargs=kwargs,
            description=description,
            enabled=enabled,
            max_instances=max_instances,
            priority=priority,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            tags=tags
        )
        
        # Thêm công việc vào manager
        return self.add_job(job)
    
    def modify_job(
        self,
        job_id: str,
        cron_expression: Optional[str] = None,
        enabled: Optional[bool] = None,
        description: Optional[str] = None,
        max_instances: Optional[int] = None,
        priority: Optional[TaskPriority] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[int] = None,
        timeout: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Sửa đổi một công việc cron.
        
        Args:
            job_id: ID của công việc cần sửa
            cron_expression: Biểu thức cron mới
            enabled: Trạng thái kích hoạt mới
            description: Mô tả mới
            max_instances: Số lượng instances tối đa mới
            priority: Độ ưu tiên mới
            max_retries: Số lần thử lại tối đa mới
            retry_delay: Thời gian chờ giữa các lần thử lại mới
            timeout: Thời gian tối đa cho phép thực thi mới
            tags: Danh sách các tag phân loại mới
            
        Returns:
            True nếu sửa đổi thành công, False nếu không tìm thấy công việc
        """
        with self._jobs_lock:
            if job_id not in self.jobs:
                self.logger.warning(f"Không tìm thấy công việc có ID {job_id} để sửa đổi")
                return False
            
            job = self.jobs[job_id]
            
            # Cập nhật các thuộc tính nếu được cung cấp
            if cron_expression is not None:
                # Validate cron expression
                try:
                    croniter(cron_expression)
                    job.cron_expression = cron_expression
                    job._update_next_run_time()
                except ValueError as e:
                    self.logger.error(f"Biểu thức cron không hợp lệ '{cron_expression}': {str(e)}")
                    return False
            
            if enabled is not None:
                job.enabled = enabled
            
            if description is not None:
                job.description = description
            
            if max_instances is not None:
                job.max_instances = max_instances
            
            if priority is not None:
                job.priority = priority
            
            if max_retries is not None:
                job.max_retries = max_retries
            
            if retry_delay is not None:
                job.retry_delay = retry_delay
            
            if timeout is not None:
                job.timeout = timeout
            
            if tags is not None:
                job.tags = tags
        
        self.logger.info(f"Đã sửa đổi công việc cron {job_id}")
        return True
    
    def remove_job(self, job_id: str) -> bool:
        """
        Xóa một công việc cron.
        
        Args:
            job_id: ID của công việc cần xóa
            
        Returns:
            True nếu xóa thành công, False nếu không tìm thấy công việc
        """
        with self._jobs_lock:
            if job_id not in self.jobs:
                self.logger.warning(f"Không tìm thấy công việc có ID {job_id} để xóa")
                return False
            
            # Xóa công việc
            del self.jobs[job_id]
            
            # Xóa task_ids đang chạy
            if job_id in self.running_task_ids:
                for task_id in self.running_task_ids[job_id]:
                    # Hủy task nếu đang chạy
                    self.task_scheduler.cancel_task(task_id)
                
                del self.running_task_ids[job_id]
        
        self.logger.info(f"Đã xóa công việc cron {job_id}")
        return True
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """
        Lấy thông tin về một công việc cron.
        
        Args:
            job_id: ID của công việc
            
        Returns:
            Dictionary chứa thông tin công việc hoặc None nếu không tìm thấy
        """
        with self._jobs_lock:
            if job_id not in self.jobs:
                return None
            
            return self.jobs[job_id].to_dict()
    
    def get_all_jobs(self) -> List[Dict]:
        """
        Lấy thông tin về tất cả các công việc cron.
        
        Returns:
            Danh sách các dictionary chứa thông tin công việc
        """
        with self._jobs_lock:
            return [job.to_dict() for job in self.jobs.values()]
    
    def enable_job(self, job_id: str) -> bool:
        """
        Kích hoạt một công việc cron.
        
        Args:
            job_id: ID của công việc
            
        Returns:
            True nếu kích hoạt thành công, False nếu không tìm thấy công việc
        """
        return self.modify_job(job_id, enabled=True)
    
    def disable_job(self, job_id: str) -> bool:
        """
        Vô hiệu hóa một công việc cron.
        
        Args:
            job_id: ID của công việc
            
        Returns:
            True nếu vô hiệu hóa thành công, False nếu không tìm thấy công việc
        """
        return self.modify_job(job_id, enabled=False)
    
    def get_running_jobs(self) -> List[Dict]:
        """
        Lấy danh sách các công việc đang chạy.
        
        Returns:
            Danh sách các dictionary chứa thông tin công việc đang chạy
        """
        with self._jobs_lock:
            return [job.to_dict() for job in self.jobs.values() 
                   if job.running_instances > 0]
    
    def load_from_config(self, config_path: Union[str, Path]) -> int:
        """
        Tải các công việc cron từ file cấu hình.
        
        Args:
            config_path: Đường dẫn đến file cấu hình
            
        Returns:
            Số lượng công việc đã tải
        """
        import json
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Format cấu hình mong đợi:
            # {
            #     "jobs": [
            #         {
            #             "job_id": "daily_backup",
            #             "cron_expression": "0 0 * * *",
            #             "function": "backup_data",
            #             "module": "real_time_inference.maintenance.backup",
            #             "args": [],
            #             "kwargs": {"backup_dir": "/path/to/backup"},
            #             "description": "Tạo backup hàng ngày vào 00:00",
            #             "enabled": true,
            #             "max_instances": 1
            #         },
            #         ...
            #     ]
            # }
            
            count = 0
            for job_config in config.get('jobs', []):
                try:
                    # Lấy module và function
                    module_name = job_config.get('module')
                    function_name = job_config.get('function')
                    
                    if not module_name or not function_name:
                        self.logger.error(f"Thiếu module hoặc function trong cấu hình: {job_config}")
                        continue
                    
                    # Import module
                    try:
                        module = __import__(module_name, fromlist=[function_name])
                        func = getattr(module, function_name)
                    except (ImportError, AttributeError) as e:
                        self.logger.error(f"Không thể import {module_name}.{function_name}: {str(e)}")
                        continue
                    
                    # Lên lịch công việc
                    self.schedule_job(
                        cron_expression=job_config.get('cron_expression'),
                        func=func,
                        args=job_config.get('args', []),
                        kwargs=job_config.get('kwargs', {}),
                        job_id=job_config.get('job_id'),
                        description=job_config.get('description', ''),
                        enabled=job_config.get('enabled', True),
                        max_instances=job_config.get('max_instances', 1),
                        priority=TaskPriority[job_config.get('priority', 'MEDIUM')],
                        max_retries=job_config.get('max_retries', 3),
                        retry_delay=job_config.get('retry_delay', 5),
                        timeout=job_config.get('timeout'),
                        tags=job_config.get('tags', [])
                    )
                    
                    count += 1
                    
                except Exception as e:
                    self.logger.error(f"Lỗi khi tải công việc từ cấu hình: {str(e)}")
            
            self.logger.info(f"Đã tải {count} công việc cron từ file cấu hình {config_path}")
            return count
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải file cấu hình {config_path}: {str(e)}")
            return 0
    
    def save_to_config(self, config_path: Union[str, Path]) -> bool:
        """
        Lưu các công việc cron vào file cấu hình.
        
        Args:
            config_path: Đường dẫn đến file cấu hình
            
        Returns:
            True nếu lưu thành công, False nếu có lỗi
        """
        import json
        
        try:
            # Tạo cấu hình
            config = {"jobs": []}
            
            with self._jobs_lock:
                for job in self.jobs.values():
                    # Lấy thông tin module và function
                    module_name = job.func.__module__
                    function_name = job.func.__name__
                    
                    # Tạo cấu hình cho công việc
                    job_config = {
                        "job_id": job.job_id,
                        "cron_expression": job.cron_expression,
                        "function": function_name,
                        "module": module_name,
                        "args": job.args,
                        "kwargs": job.kwargs,
                        "description": job.description,
                        "enabled": job.enabled,
                        "max_instances": job.max_instances,
                        "priority": job.priority.name,
                        "max_retries": job.max_retries,
                        "retry_delay": job.retry_delay,
                        "timeout": job.timeout,
                        "tags": job.tags
                    }
                    
                    config["jobs"].append(job_config)
            
            # Lưu cấu hình
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4, default=str)
            
            self.logger.info(f"Đã lưu {len(config['jobs'])} công việc cron vào file cấu hình {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu file cấu hình {config_path}: {str(e)}")
            return False

# Tạo singleton instance cho toàn bộ ứng dụng
_cron_manager_instance = None

def get_cron_manager(task_scheduler: Optional[TaskScheduler] = None, check_interval: float = 60.0, logger: Optional[logging.Logger] = None):
    """
    Lấy instance toàn cục của CronJobManager (singleton pattern).
    
    Args:
        task_scheduler: Task scheduler để lên lịch tác vụ (tùy chọn)
        check_interval: Khoảng thời gian kiểm tra các công việc (giây)
        logger: Logger tùy chỉnh
        
    Returns:
        Instance của CronJobManager
    """
    global _cron_manager_instance
    if _cron_manager_instance is None:
        _cron_manager_instance = CronJobManager(
            task_scheduler=task_scheduler,
            check_interval=check_interval,
            logger=logger
        )
    return _cron_manager_instance


def cron(cron_expression: str, **job_kwargs):
    """
    Decorator để lên lịch một hàm theo cú pháp cron.
    
    Args:
        cron_expression: Biểu thức cron (ví dụ: "0 * * * *" cho mỗi giờ)
        **job_kwargs: Các tham số khác cho công việc
        
    Returns:
        Decorator
    """
    def decorator(func):
        # Lấy CronJobManager instance
        cron_manager = get_cron_manager()
        
        # Lên lịch công việc
        job_id = cron_manager.schedule_job(
            cron_expression=cron_expression,
            func=func,
            **job_kwargs
        )
        
        # Thêm thuộc tính job_id vào hàm
        func.job_id = job_id
        
        return func
    
    return decorator


# Các utility functions

def parse_cron_expression(expression: str) -> Dict:
    """
    Phân tích biểu thức cron thành mô tả ngôn ngữ tự nhiên.
    
    Args:
        expression: Biểu thức cron
        
    Returns:
        Dictionary mô tả biểu thức
    """
    parts = expression.split()
    if len(parts) < 5:
        raise ValueError(f"Biểu thức cron không hợp lệ: {expression}")
    
    minute, hour, day, month, weekday = parts[:5]
    result = {
        "original": expression,
        "minute": minute,
        "hour": hour,
        "day": day,
        "month": month,
        "weekday": weekday,
        "description": ""
    }
    
    # Tạo mô tả
    description = []
    
    # Phần weekday
    weekday_names = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
    if weekday != "*":
        if weekday.isdigit():
            day_index = int(weekday) % 7
            description.append(f"vào {weekday_names[day_index]}")
        elif "," in weekday:
            days = [weekday_names[int(d) % 7] for d in weekday.split(",") if d.isdigit()]
            description.append(f"vào {', '.join(days)}")
        elif "-" in weekday:
            start, end = weekday.split("-")
            if start.isdigit() and end.isdigit():
                start_index = int(start) % 7
                end_index = int(end) % 7
                description.append(f"từ {weekday_names[start_index]} đến {weekday_names[end_index]}")
    
    # Phần month
    month_names = ["", "Tháng 1", "Tháng 2", "Tháng 3", "Tháng 4", "Tháng 5", "Tháng 6", 
                  "Tháng 7", "Tháng 8", "Tháng 9", "Tháng 10", "Tháng 11", "Tháng 12"]
    if month != "*":
        if month.isdigit():
            description.append(f"trong {month_names[int(month)]}")
        elif "," in month:
            months = [month_names[int(m)] for m in month.split(",") if m.isdigit()]
            description.append(f"trong {', '.join(months)}")
        elif "-" in month:
            start, end = month.split("-")
            if start.isdigit() and end.isdigit():
                description.append(f"từ {month_names[int(start)]} đến {month_names[int(end)]}")
    
    # Phần day
    if day != "*":
        if day.isdigit():
            description.append(f"vào ngày {day}")
        elif "," in day:
            days = day.split(",")
            description.append(f"vào ngày {', '.join(days)}")
        elif "-" in day:
            start, end = day.split("-")
            description.append(f"từ ngày {start} đến ngày {end}")
    
    # Phần hour và minute
    time_desc = []
    if hour != "*":
        if hour.isdigit():
            time_desc.append(f"{hour} giờ")
        elif "," in hour:
            hours = hour.split(",")
            time_desc.append(f"các giờ {', '.join(hours)}")
        elif "-" in hour:
            start, end = hour.split("-")
            time_desc.append(f"từ {start} đến {end} giờ")
        elif "/" in hour:
            _, step = hour.split("/")
            time_desc.append(f"mỗi {step} giờ")
    else:
        time_desc.append("mỗi giờ")
    
    if minute != "*":
        if minute.isdigit():
            time_desc.append(f"{minute} phút")
        elif "," in minute:
            minutes = minute.split(",")
            time_desc.append(f"phút {', '.join(minutes)}")
        elif "-" in minute:
            start, end = minute.split("-")
            time_desc.append(f"từ phút {start} đến phút {end}")
        elif "/" in minute:
            _, step = minute.split("/")
            time_desc.append(f"mỗi {step} phút")
    else:
        if hour != "*" and hour.isdigit():
            time_desc.append("00 phút")
        else:
            time_desc.append("mỗi phút")
    
    description.append(" ".join(time_desc))
    
    # Kết hợp mô tả
    result["description"] = "Chạy " + ", ".join(description)
    
    return result


def get_next_run_times(cron_expression: str, count: int = 5) -> List[datetime]:
    """
    Tính toán các thời điểm chạy tiếp theo cho một biểu thức cron.
    
    Args:
        cron_expression: Biểu thức cron
        count: Số lượng thời điểm muốn lấy
        
    Returns:
        Danh sách các thời điểm chạy tiếp theo
    """
    try:
        base_time = datetime.now()
        cron = croniter(cron_expression, base_time)
        
        result = []
        for _ in range(count):
            next_time = cron.get_next(datetime)
            result.append(next_time)
        
        return result
    except ValueError as e:
        raise ValueError(f"Biểu thức cron không hợp lệ '{cron_expression}': {str(e)}")


def convert_to_cron(
    minute: Union[str, int, List[int]] = "*",
    hour: Union[str, int, List[int]] = "*",
    day: Union[str, int, List[int]] = "*",
    month: Union[str, int, List[int]] = "*",
    weekday: Union[str, int, List[int]] = "*"
) -> str:
    """
    Chuyển đổi từ các tham số riêng biệt thành biểu thức cron.
    
    Args:
        minute: Phút (0-59)
        hour: Giờ (0-23)
        day: Ngày (1-31)
        month: Tháng (1-12)
        weekday: Ngày trong tuần (0-6, 0=Thứ Hai)
        
    Returns:
        Biểu thức cron
    """
    # Xử lý danh sách
    def process_param(param):
        if isinstance(param, list):
            return ",".join(str(p) for p in param)
        return str(param)
    
    minute = process_param(minute)
    hour = process_param(hour)
    day = process_param(day)
    month = process_param(month)
    weekday = process_param(weekday)
    
    return f"{minute} {hour} {day} {month} {weekday}"