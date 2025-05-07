"""
Cập nhật mô hình tự động.
File này định nghĩa lớp ModelUpdater để quản lý quá trình tái huấn luyện,
cập nhật và triển khai các mô hình tự động dựa trên hiệu suất.
"""

import os
import sys
import time
import logging
import threading
import json
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from models.training_pipeline.trainer import Trainer
from models.agents.base_agent import BaseAgent
from models.agents.dqn_agent import DQNAgent
from backtesting.performance_metrics import PerformanceMetrics
from environments.base_environment import BaseEnvironment
from environments.trading_gym.trading_env import TradingEnv


class ModelUpdater:
    """
    Quản lý việc tái huấn luyện và cập nhật các mô hình tự động.
    Cung cấp các phương thức để theo dõi hiệu suất, tái huấn luyện khi cần thiết,
    và triển khai các mô hình đã cập nhật.
    """
    
    def __init__(
        self,
        retraining_interval: float = 86400.0,  # 24 giờ
        performance_threshold: float = -0.05,
        improvement_threshold: float = 0.02,
        max_retraining_attempts: int = 3,
        max_concurrent_retraining: int = 2,
        models_dir: Optional[Union[str, Path]] = None,
        auto_deploy: bool = True,
        keep_history: int = 5,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Khởi tạo ModelUpdater.
        
        Args:
            retraining_interval: Khoảng thời gian (giây) giữa các lần tái huấn luyện
            performance_threshold: Ngưỡng hiệu suất để kích hoạt tái huấn luyện
            improvement_threshold: Ngưỡng cải thiện tối thiểu để chấp nhận mô hình mới
            max_retraining_attempts: Số lần thử tái huấn luyện tối đa
            max_concurrent_retraining: Số lượng tái huấn luyện đồng thời tối đa
            models_dir: Thư mục lưu trữ mô hình
            auto_deploy: Tự động triển khai mô hình mới
            keep_history: Số lượng phiên bản mô hình giữ lại
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("model_updater")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Lưu trữ các tham số
        self.retraining_interval = retraining_interval
        self.performance_threshold = performance_threshold
        self.improvement_threshold = improvement_threshold
        self.max_retraining_attempts = max_retraining_attempts
        self.max_concurrent_retraining = max_concurrent_retraining
        self.auto_deploy = auto_deploy
        self.keep_history = keep_history
        self.kwargs = kwargs
        
        # Thiết lập thư mục mô hình
        if models_dir is None:
            self.models_dir = Path(self.system_config.get("MODELS_DIR", "./models"))
        else:
            self.models_dir = Path(models_dir)
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Hàng đợi và luồng tái huấn luyện
        self.retraining_queue = queue.PriorityQueue()
        self.retraining_pool = ThreadPoolExecutor(max_workers=max_concurrent_retraining)
        self.retraining_thread = None
        self.running = False
        
        # Khóa đồng bộ
        self.queue_lock = threading.RLock()
        
        # Theo dõi mô hình
        self.model_status = {}  # Dict {model_id: status_info}
        self.retraining_history = {}  # Dict {model_id: [history_info, ...]}
        self.performance_cache = {}  # Dict {model_id: {timestamp: performance}}
        self.active_retraining = set()  # Set các model_id đang được tái huấn luyện
        
        # Các cấu hình huấn luyện mặc định
        self.default_configs = {
            'dqn': {
                'batch_size': 64,
                'memory_size': 10000,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'epsilon': 1.0,
                'epsilon_min': 0.01,
                'epsilon_decay': 0.995,
                'hidden_layers': [64, 64],
                'update_target_freq': 100,
                'num_episodes': 500
            },
            'ppo': {
                'batch_size': 64,
                'optimizer_lr': 0.0003,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_ratio': 0.2,
                'value_coef': 0.5,
                'entropy_coef': 0.01,
                'num_epochs': 10,
                'num_episodes': 300
            }
        }
        
        self.logger.info(f"Đã khởi tạo ModelUpdater với {max_concurrent_retraining} luồng tái huấn luyện tối đa")
    
    def start(self) -> bool:
        """
        Bắt đầu luồng tái huấn luyện tự động.
        
        Returns:
            bool: True nếu khởi động thành công, False nếu không
        """
        if self.running:
            self.logger.warning("ModelUpdater đã đang chạy")
            return False
        
        # Đặt trạng thái chạy
        self.running = True
        
        # Tải trạng thái mô hình nếu có
        self._load_model_status()
        
        # Khởi động thread kiểm tra hàng đợi
        self.retraining_thread = threading.Thread(target=self._retraining_loop, daemon=True)
        self.retraining_thread.start()
        
        self.logger.info("ModelUpdater đã khởi động")
        return True
    
    def stop(self) -> bool:
        """
        Dừng luồng tái huấn luyện tự động.
        
        Returns:
            bool: True nếu dừng thành công, False nếu không
        """
        if not self.running:
            self.logger.warning("ModelUpdater không chạy")
            return False
        
        # Đặt trạng thái dừng
        self.running = False
        
        # Đợi thread kiểm tra hàng đợi kết thúc
        if self.retraining_thread and self.retraining_thread.is_alive():
            self.retraining_thread.join(timeout=10.0)
        
        # Lưu trạng thái mô hình
        self._save_model_status()
        
        # Đóng thread pool
        self.retraining_pool.shutdown(wait=True)
        
        self.logger.info("ModelUpdater đã dừng")
        return True
    
    def schedule_retraining(
        self, 
        model_id: str, 
        model_info: Dict[str, Any], 
        priority: int = 1, 
        delay: float = 0.0
    ) -> bool:
        """
        Lập lịch tái huấn luyện mô hình.
        
        Args:
            model_id: ID mô hình
            model_info: Thông tin mô hình
            priority: Mức độ ưu tiên (thấp hơn = ưu tiên cao hơn)
            delay: Độ trễ (giây) trước khi tái huấn luyện
            
        Returns:
            bool: True nếu lập lịch thành công, False nếu không
        """
        # Kiểm tra xem mô hình có đang được tái huấn luyện không
        if model_id in self.active_retraining:
            self.logger.warning(f"Mô hình '{model_id}' đã đang được tái huấn luyện")
            return False
        
        # Tính thời gian tái huấn luyện
        retraining_time = time.time() + delay
        
        # Tạo mục hàng đợi
        queue_item = (priority, retraining_time, model_id, model_info)
        
        # Thêm vào hàng đợi
        with self.queue_lock:
            self.retraining_queue.put(queue_item)
            
            # Cập nhật trạng thái mô hình
            if model_id not in self.model_status:
                self.model_status[model_id] = {
                    'id': model_id,
                    'last_updated': datetime.now().isoformat(),
                    'retraining_count': 0,
                    'current_version': 1,
                    'status': 'scheduled'
                }
            else:
                self.model_status[model_id]['status'] = 'scheduled'
                self.model_status[model_id]['scheduled_time'] = retraining_time
        
        self.logger.info(
            f"Đã lập lịch tái huấn luyện mô hình '{model_id}' với ưu tiên {priority}, "
            f"độ trễ {delay:.1f}s"
        )
        
        return True
    
    def retrain_model(
        self, 
        model_info: Dict[str, Any], 
        force: bool = False,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Tái huấn luyện mô hình ngay lập tức.
        
        Args:
            model_info: Thông tin mô hình
            force: Bắt buộc tái huấn luyện ngay cả khi mô hình đang hoạt động tốt
            custom_config: Cấu hình tái huấn luyện tùy chỉnh
            
        Returns:
            Dict: Thông tin mô hình đã cập nhật
        """
        model_id = model_info.get('id')
        
        if not model_id:
            model_id = f"model_{int(time.time())}"
            model_info['id'] = model_id
        
        # Kiểm tra xem mô hình có đang được tái huấn luyện không
        if model_id in self.active_retraining:
            self.logger.warning(f"Mô hình '{model_id}' đã đang được tái huấn luyện")
            raise RuntimeError(f"Mô hình '{model_id}' đã đang được tái huấn luyện")
        
        # Đánh dấu mô hình đang được tái huấn luyện
        self.active_retraining.add(model_id)
        
        try:
            # Cập nhật trạng thái mô hình
            if model_id not in self.model_status:
                self.model_status[model_id] = {
                    'id': model_id,
                    'last_updated': datetime.now().isoformat(),
                    'retraining_count': 0,
                    'current_version': 1,
                    'status': 'retraining'
                }
            else:
                self.model_status[model_id]['status'] = 'retraining'
                self.model_status[model_id]['retraining_start'] = datetime.now().isoformat()
            
            # Thực hiện tái huấn luyện
            updated_model_info = self._perform_retraining(model_info, force, custom_config)
            
            # Cập nhật trạng thái mô hình
            if model_id in self.model_status:
                self.model_status[model_id]['status'] = 'active'
                self.model_status[model_id]['last_updated'] = datetime.now().isoformat()
                self.model_status[model_id]['retraining_count'] += 1
                self.model_status[model_id]['current_version'] += 1
                
                # Thêm lịch sử tái huấn luyện
                if model_id not in self.retraining_history:
                    self.retraining_history[model_id] = []
                
                self.retraining_history[model_id].append({
                    'timestamp': datetime.now().isoformat(),
                    'version': self.model_status[model_id]['current_version'],
                    'performance_improvement': updated_model_info.get('performance_improvement', 0.0),
                    'training_time': updated_model_info.get('training_time', 0.0),
                    'force_retrained': force
                })
                
                # Giới hạn kích thước lịch sử
                if len(self.retraining_history[model_id]) > 100:
                    self.retraining_history[model_id] = self.retraining_history[model_id][-100:]
            
            # Lưu trạng thái
            self._save_model_status()
            
            self.logger.info(f"Đã hoàn thành tái huấn luyện mô hình '{model_id}'")
            
            return updated_model_info
            
        except Exception as e:
            # Cập nhật trạng thái mô hình nếu có lỗi
            if model_id in self.model_status:
                self.model_status[model_id]['status'] = 'error'
                self.model_status[model_id]['last_error'] = str(e)
                self.model_status[model_id]['error_time'] = datetime.now().isoformat()
            
            self.logger.error(f"Lỗi khi tái huấn luyện mô hình '{model_id}': {str(e)}")
            raise
            
        finally:
            # Xóa khỏi danh sách đang tái huấn luyện
            self.active_retraining.discard(model_id)
    
    def update_model_performance(
        self, 
        model_id: str, 
        performance: float, 
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Cập nhật hiệu suất mô hình.
        
        Args:
            model_id: ID mô hình
            performance: Giá trị hiệu suất
            timestamp: Thời gian đo hiệu suất
            
        Returns:
            bool: True nếu cập nhật thành công và mô hình cần được tái huấn luyện
            
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        timestamp_str = timestamp.isoformat()
        
        # Thêm vào cache hiệu suất
        if model_id not in self.performance_cache:
            self.performance_cache[model_id] = {}
        
        self.performance_cache[model_id][timestamp_str] = performance
        
        # Cập nhật trạng thái mô hình
        if model_id in self.model_status:
            self.model_status[model_id]['last_performance'] = performance
            self.model_status[model_id]['last_performance_time'] = timestamp_str
            
            # Tính hiệu suất trung bình
            performances = list(self.performance_cache[model_id].values())
            if len(performances) > 0:
                avg_performance = sum(performances) / len(performances)
                self.model_status[model_id]['avg_performance'] = avg_performance
        
        # Kiểm tra xem có cần tái huấn luyện không
        needs_retraining = performance < self.performance_threshold
        
        # Kiểm tra xem mô hình có đang được lập lịch tái huấn luyện không
        if needs_retraining and model_id not in self.active_retraining:
            # Kiểm tra thời gian tái huấn luyện gần nhất
            if model_id in self.model_status:
                last_retrained_str = self.model_status[model_id].get('last_retrained')
                
                if last_retrained_str:
                    try:
                        last_retrained = datetime.fromisoformat(last_retrained_str)
                        time_since_last = (datetime.now() - last_retrained).total_seconds()
                        
                        # Chỉ tái huấn luyện nếu đã qua đủ thời gian
                        if time_since_last < self.retraining_interval:
                            needs_retraining = False
                    except (ValueError, TypeError):
                        pass
        
        self.logger.debug(
            f"Đã cập nhật hiệu suất mô hình '{model_id}': {performance:.4f}, "
            f"cần tái huấn luyện: {needs_retraining}"
        )
        
        return needs_retraining
    
    def get_model_status(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy trạng thái mô hình.
        
        Args:
            model_id: ID mô hình
            
        Returns:
            Dict: Thông tin trạng thái hoặc None nếu không tìm thấy
        """
        return self.model_status.get(model_id)
    
    def get_retraining_history(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử tái huấn luyện mô hình.
        
        Args:
            model_id: ID mô hình
            
        Returns:
            List: Lịch sử tái huấn luyện
        """
        return self.retraining_history.get(model_id, [])
    
    def get_all_models_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Lấy trạng thái của tất cả mô hình.
        
        Returns:
            Dict: {model_id: status_info}
        """
        return self.model_status.copy()
    
    def deploy_model(
        self, 
        model_id: str, 
        source_path: Optional[Union[str, Path]] = None,
        target_path: Optional[Union[str, Path]] = None,
        version: Optional[int] = None
    ) -> bool:
        """
        Triển khai mô hình vào môi trường sản xuất.
        
        Args:
            model_id: ID mô hình
            source_path: Đường dẫn nguồn
            target_path: Đường dẫn đích
            version: Phiên bản cần triển khai
            
        Returns:
            bool: True nếu triển khai thành công, False nếu không
        """
        # Nếu không có source_path, tìm mô hình dựa trên ID và phiên bản
        if source_path is None:
            # Xác định phiên bản
            if version is None and model_id in self.model_status:
                version = self.model_status[model_id].get('current_version', 1)
            
            version_str = f"v{version}" if version else "latest"
            source_path = self.models_dir / model_id / version_str
        else:
            source_path = Path(source_path)
        
        # Nếu không có target_path, sử dụng thư mục triển khai mặc định
        if target_path is None:
            target_path = self.models_dir / model_id / "production"
        else:
            target_path = Path(target_path)
        
        try:
            # Kiểm tra xem source_path có tồn tại không
            if not source_path.exists():
                self.logger.error(f"Không tìm thấy mô hình nguồn tại {source_path}")
                return False
            
            # Tạo thư mục đích nếu chưa tồn tại
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Nếu target_path đã tồn tại, tạo bản sao lưu
            if target_path.exists():
                backup_path = target_path.parent / f"{target_path.name}_backup_{int(time.time())}"
                shutil.copytree(target_path, backup_path)
                self.logger.info(f"Đã tạo bản sao lưu mô hình cũ tại {backup_path}")
            
            # Xóa thư mục đích nếu đã tồn tại
            if target_path.exists():
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()
            
            # Sao chép mô hình
            shutil.copytree(source_path, target_path)
            
            # Cập nhật trạng thái mô hình
            if model_id in self.model_status:
                self.model_status[model_id]['deployed_version'] = version
                self.model_status[model_id]['deployed_time'] = datetime.now().isoformat()
                self.model_status[model_id]['deployment_path'] = str(target_path)
            
            self.logger.info(f"Đã triển khai mô hình '{model_id}' từ {source_path} đến {target_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi triển khai mô hình '{model_id}': {str(e)}")
            return False
    
    def compare_models(
        self, 
        model_id: str, 
        old_version: Optional[int] = None, 
        new_version: Optional[int] = None,
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        So sánh hiệu suất của hai phiên bản mô hình.
        
        Args:
            model_id: ID mô hình
            old_version: Phiên bản cũ (None để sử dụng phiên bản hiện tại)
            new_version: Phiên bản mới (None để sử dụng phiên bản mới nhất)
            evaluation_config: Cấu hình đánh giá
            
        Returns:
            Dict: Kết quả so sánh
        """
        # Xác định phiên bản cũ
        if old_version is None and model_id in self.model_status:
            if 'deployed_version' in self.model_status[model_id]:
                old_version = self.model_status[model_id]['deployed_version']
            else:
                old_version = self.model_status[model_id].get('current_version', 1) - 1
        
        # Xác định phiên bản mới
        if new_version is None and model_id in self.model_status:
            new_version = self.model_status[model_id].get('current_version', 1)
        
        # Kiểm tra xem phiên bản cũ và mới có tồn tại không
        old_path = self.models_dir / model_id / f"v{old_version}"
        new_path = self.models_dir / model_id / f"v{new_version}"
        
        if not old_path.exists():
            raise FileNotFoundError(f"Không tìm thấy phiên bản cũ tại {old_path}")
        
        if not new_path.exists():
            raise FileNotFoundError(f"Không tìm thấy phiên bản mới tại {new_path}")
        
        # Tải mô hình
        old_model_info = self._load_model_metadata(old_path)
        new_model_info = self._load_model_metadata(new_path)
        
        # Tải và khởi tạo mô hình
        old_model = self._load_model(old_path, old_model_info)
        new_model = self._load_model(new_path, new_model_info)
        
        # Thiết lập môi trường đánh giá
        eval_env = self._create_evaluation_environment(model_id, evaluation_config)
        
        # Thực hiện đánh giá
        old_metrics = self._evaluate_model(old_model, eval_env, old_model_info)
        new_metrics = self._evaluate_model(new_model, eval_env, new_model_info)
        
        # Tính sự cải thiện
        improvements = {}
        for key in set(old_metrics.keys()) | set(new_metrics.keys()):
            if key in old_metrics and key in new_metrics:
                old_value = old_metrics[key]
                new_value = new_metrics[key]
                
                if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                    abs_improvement = new_value - old_value
                    rel_improvement = abs_improvement / abs(old_value) if abs(old_value) > 1e-10 else float('inf')
                    
                    improvements[key] = {
                        'old_value': old_value,
                        'new_value': new_value,
                        'absolute_improvement': abs_improvement,
                        'relative_improvement': rel_improvement
                    }
        
        # Tính điểm cải thiện tổng thể
        if 'total_return' in improvements:
            overall_improvement = improvements['total_return']['relative_improvement']
        elif 'sharpe_ratio' in improvements:
            overall_improvement = improvements['sharpe_ratio']['relative_improvement']
        else:
            # Tính trung bình cải thiện các chỉ số
            rel_improvements = [imp['relative_improvement'] for imp in improvements.values() 
                              if isinstance(imp, dict) and 'relative_improvement' in imp]
            
            overall_improvement = sum(rel_improvements) / len(rel_improvements) if rel_improvements else 0.0
        
        # Tạo kết quả so sánh
        comparison_result = {
            'model_id': model_id,
            'old_version': old_version,
            'new_version': new_version,
            'comparison_time': datetime.now().isoformat(),
            'old_metrics': old_metrics,
            'new_metrics': new_metrics,
            'improvements': improvements,
            'overall_improvement': overall_improvement,
            'recommendation': 'deploy' if overall_improvement > self.improvement_threshold else 'keep_current'
        }
        
        self.logger.info(
            f"So sánh mô hình '{model_id}': v{old_version} vs v{new_version}, "
            f"cải thiện: {overall_improvement:.4f}, khuyến nghị: {comparison_result['recommendation']}"
        )
        
        return comparison_result
    
    def cleanup_old_models(self, model_id: str, keep_versions: int = None) -> int:
        """
        Dọn dẹp các phiên bản cũ của mô hình.
        
        Args:
            model_id: ID mô hình
            keep_versions: Số phiên bản giữ lại (None để sử dụng giá trị mặc định)
            
        Returns:
            int: Số phiên bản đã xóa
        """
        if keep_versions is None:
            keep_versions = self.keep_history
        
        # Kiểm tra thư mục mô hình
        model_dir = self.models_dir / model_id
        
        if not model_dir.exists() or not model_dir.is_dir():
            self.logger.warning(f"Không tìm thấy thư mục mô hình '{model_id}'")
            return 0
        
        # Tìm tất cả phiên bản
        versions = []
        
        for item in model_dir.iterdir():
            if item.is_dir() and item.name.startswith('v'):
                try:
                    version = int(item.name[1:])
                    versions.append((version, item))
                except ValueError:
                    pass
        
        # Sắp xếp theo phiên bản giảm dần
        versions.sort(reverse=True)
        
        # Xác định phiên bản được triển khai
        deployed_version = None
        if model_id in self.model_status and 'deployed_version' in self.model_status[model_id]:
            deployed_version = self.model_status[model_id]['deployed_version']
        
        # Xóa các phiên bản cũ
        deleted_count = 0
        
        for i, (version, path) in enumerate(versions):
            # Bỏ qua phiên bản mới và phiên bản được triển khai
            if i < keep_versions or version == deployed_version:
                continue
            
            try:
                shutil.rmtree(path)
                deleted_count += 1
                
                self.logger.info(f"Đã xóa phiên bản cũ v{version} của mô hình '{model_id}'")
            except Exception as e:
                self.logger.error(f"Lỗi khi xóa phiên bản v{version} của mô hình '{model_id}': {str(e)}")
        
        return deleted_count
    
    def _retraining_loop(self) -> None:
        """
        Vòng lặp kiểm tra hàng đợi tái huấn luyện.
        """
        self.logger.info("Bắt đầu vòng lặp tái huấn luyện")
        
        while self.running:
            try:
                # Kiểm tra số lượng tái huấn luyện đang thực hiện
                active_count = len(self.active_retraining)
                
                if active_count >= self.max_concurrent_retraining:
                    # Đợi nếu đã đạt giới hạn
                    time.sleep(5.0)
                    continue
                
                # Lấy mô hình tiếp theo từ hàng đợi (không chặn)
                try:
                    priority, scheduled_time, model_id, model_info = self.retraining_queue.get(block=False)
                    
                    # Kiểm tra xem đã đến thời gian chưa
                    current_time = time.time()
                    
                    if current_time < scheduled_time:
                        # Chưa đến thời gian, đưa trở lại hàng đợi
                        self.retraining_queue.put((priority, scheduled_time, model_id, model_info))
                        time.sleep(1.0)
                        continue
                    
                    # Kiểm tra xem mô hình có đang được tái huấn luyện không
                    if model_id in self.active_retraining:
                        self.logger.warning(f"Mô hình '{model_id}' đã đang được tái huấn luyện, bỏ qua")
                        self.retraining_queue.task_done()
                        continue
                    
                    # Thực hiện tái huấn luyện trong thread pool
                    self.retraining_pool.submit(self._retraining_task, model_id, model_info)
                    
                    # Đánh dấu task đã xử lý
                    self.retraining_queue.task_done()
                    
                except queue.Empty:
                    # Không có mô hình nào trong hàng đợi, đợi một chút
                    time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Lỗi trong vòng lặp tái huấn luyện: {str(e)}")
                time.sleep(5.0)
        
        self.logger.info("Kết thúc vòng lặp tái huấn luyện")
    
    def _retraining_task(self, model_id: str, model_info: Dict[str, Any]) -> None:
        """
        Task tái huấn luyện mô hình trong thread riêng.
        
        Args:
            model_id: ID mô hình
            model_info: Thông tin mô hình
        """
        # Đánh dấu mô hình đang được tái huấn luyện
        self.active_retraining.add(model_id)
        
        try:
            # Cập nhật trạng thái mô hình
            if model_id in self.model_status:
                self.model_status[model_id]['status'] = 'retraining'
                self.model_status[model_id]['retraining_start'] = datetime.now().isoformat()
            
            # Thực hiện tái huấn luyện
            updated_model_info = self._perform_retraining(model_info)
            
            # Nếu tái huấn luyện thành công và cấu hình auto_deploy
            if self.auto_deploy and updated_model_info.get('performance_improvement', 0.0) > self.improvement_threshold:
                # Triển khai mô hình mới
                new_version = self.model_status[model_id]['current_version'] if model_id in self.model_status else 1
                self.deploy_model(model_id, version=new_version)
            
            # Cập nhật trạng thái mô hình
            if model_id in self.model_status:
                self.model_status[model_id]['status'] = 'active'
                self.model_status[model_id]['last_updated'] = datetime.now().isoformat()
                self.model_status[model_id]['last_retrained'] = datetime.now().isoformat()
                self.model_status[model_id]['retraining_count'] += 1
            
            # Dọn dẹp các phiên bản cũ
            self.cleanup_old_models(model_id)
            
            # Lưu trạng thái
            self._save_model_status()
            
            self.logger.info(f"Đã hoàn thành tái huấn luyện mô hình '{model_id}'")
            
        except Exception as e:
            # Cập nhật trạng thái mô hình nếu có lỗi
            if model_id in self.model_status:
                self.model_status[model_id]['status'] = 'error'
                self.model_status[model_id]['last_error'] = str(e)
                self.model_status[model_id]['error_time'] = datetime.now().isoformat()
            
            self.logger.error(f"Lỗi khi tái huấn luyện mô hình '{model_id}': {str(e)}")
            
        finally:
            # Xóa khỏi danh sách đang tái huấn luyện
            self.active_retraining.discard(model_id)
    
    def _perform_retraining(
        self, 
        model_info: Dict[str, Any], 
        force: bool = False,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Thực hiện tái huấn luyện mô hình.
        
        Args:
            model_info: Thông tin mô hình
            force: Bắt buộc tái huấn luyện ngay cả khi mô hình đang hoạt động tốt
            custom_config: Cấu hình tái huấn luyện tùy chỉnh
            
        Returns:
            Dict: Thông tin mô hình đã cập nhật
        """
        model_id = model_info.get('id', f"model_{int(time.time())}")
        model_type = model_info.get('type', 'dqn')
        agent_path = model_info.get('agent_path')
        
        # Tạo thư mục lưu trữ cho mô hình
        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Xác định phiên bản mới
        if model_id in self.model_status:
            new_version = self.model_status[model_id].get('current_version', 0) + 1
        else:
            new_version = 1
        
        version_dir = model_dir / f"v{new_version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        try:
            # Tải mô hình hiện tại nếu có
            current_model = None
            current_performance = 0.0
            
            if agent_path and Path(agent_path).exists():
                try:
                    current_model = self._load_model(Path(agent_path), model_info)
                    
                    # Đánh giá hiệu suất hiện tại
                    eval_env = self._create_evaluation_environment(model_id, custom_config)
                    current_metrics = self._evaluate_model(current_model, eval_env, model_info)
                    current_performance = current_metrics.get('total_return', 0.0)
                    
                    self.logger.info(f"Hiệu suất hiện tại của mô hình '{model_id}': {current_performance:.4f}")
                    
                    # Kiểm tra xem có cần tái huấn luyện không
                    if not force and current_performance > self.performance_threshold:
                        self.logger.info(
                            f"Mô hình '{model_id}' đang hoạt động tốt ({current_performance:.4f} > {self.performance_threshold}), "
                            f"bỏ qua tái huấn luyện"
                        )
                        
                        # Vẫn cập nhật trạng thái mô hình
                        if model_id in self.model_status:
                            self.model_status[model_id]['status'] = 'active'
                            self.model_status[model_id]['last_evaluation'] = datetime.now().isoformat()
                            self.model_status[model_id]['last_performance'] = current_performance
                        
                        # Sao chép mô hình hiện tại
                        shutil.copytree(Path(agent_path).parent, version_dir, dirs_exist_ok=True)
                        
                        # Trả về thông tin mô hình
                        return {
                            'id': model_id,
                            'type': model_type,
                            'version': new_version,
                            'performance': current_performance,
                            'performance_improvement': 0.0,
                            'training_time': 0.0,
                            'retrained': False,
                            'agent_path': str(version_dir / Path(agent_path).name)
                        }
                        
                except Exception as e:
                    self.logger.warning(f"Lỗi khi tải mô hình hiện tại '{model_id}': {str(e)}, sẽ tái huấn luyện từ đầu")
            
            # Thiết lập cấu hình huấn luyện
            training_config = self._get_training_config(model_type, model_info, custom_config)
            
            # Tạo môi trường huấn luyện
            train_env = self._create_training_environment(model_id, training_config)
            
            # Tạo agent mới
            agent = self._create_agent(model_type, train_env, training_config)
            
            # Tạo trainer
            trainer = Trainer(
                agent=agent,
                env=train_env,
                config=training_config,
                output_dir=version_dir,
                experiment_name=f"{model_id}_v{new_version}",
                logger=self.logger
            )
            
            # Huấn luyện mô hình
            self.logger.info(f"Bắt đầu huấn luyện mô hình '{model_id}' phiên bản {new_version}")
            history = trainer.train()
            
            # Lưu mô hình
            save_path = trainer.save_agent(is_best=True)
            
            # Đánh giá mô hình mới
            eval_env = self._create_evaluation_environment(model_id, custom_config)
            new_metrics = self._evaluate_model(agent, eval_env, model_info)
            new_performance = new_metrics.get('total_return', 0.0)
            
            # Tính cải thiện hiệu suất
            performance_improvement = new_performance - current_performance
            
            # Lưu metadata
            metadata = {
                'id': model_id,
                'type': model_type,
                'version': new_version,
                'training_config': training_config,
                'history': history,
                'metrics': new_metrics,
                'previous_performance': current_performance,
                'performance': new_performance,
                'performance_improvement': performance_improvement,
                'training_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'agent_path': save_path
            }
            
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            self.logger.info(
                f"Đã hoàn thành huấn luyện mô hình '{model_id}' phiên bản {new_version}, "
                f"hiệu suất: {new_performance:.4f} (cải thiện: {performance_improvement:.4f})"
            )
            
            # Trả về thông tin mô hình
            return {
                'id': model_id,
                'type': model_type,
                'version': new_version,
                'performance': new_performance,
                'performance_improvement': performance_improvement,
                'training_time': time.time() - start_time,
                'retrained': True,
                'agent_path': save_path
            }
            
        except Exception as e:
            # Xử lý lỗi
            self.logger.error(f"Lỗi khi huấn luyện mô hình '{model_id}': {str(e)}")
            
            # Xóa thư mục phiên bản nếu có lỗi
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            raise
    
    def _get_training_config(
        self, 
        model_type: str, 
        model_info: Dict[str, Any],
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Lấy cấu hình huấn luyện cho mô hình.
        
        Args:
            model_type: Loại mô hình
            model_info: Thông tin mô hình
            custom_config: Cấu hình tùy chỉnh
            
        Returns:
            Dict: Cấu hình huấn luyện
        """
        # Lấy cấu hình mặc định cho loại mô hình
        if model_type.lower() in self.default_configs:
            config = self.default_configs[model_type.lower()].copy()
        else:
            config = self.default_configs['dqn'].copy()
        
        # Cập nhật từ model_info
        if 'training_config' in model_info:
            config.update(model_info['training_config'])
        
        # Cập nhật từ custom_config
        if custom_config:
            config.update(custom_config)
        
        # Điều chỉnh cấu hình tái huấn luyện
        if 'num_episodes' in config:
            # Giảm số lượng episode cho tái huấn luyện
            config['num_episodes'] = int(config['num_episodes'] * 0.6)
        
        # Thêm timestamp
        config['timestamp'] = datetime.now().isoformat()
        
        return config
    
    def _create_training_environment(
        self, 
        model_id: str, 
        config: Dict[str, Any]
    ) -> BaseEnvironment:
        """
        Tạo môi trường huấn luyện.
        
        Args:
            model_id: ID mô hình
            config: Cấu hình huấn luyện
            
        Returns:
            BaseEnvironment: Môi trường huấn luyện
        """
        # Lấy cấu hình môi trường
        env_config = config.get('environment', {})
        env_type = env_config.get('type', 'trading')
        
        # Tạo môi trường dựa trên loại
        if env_type == 'trading':
            # Tạo môi trường giao dịch
            data_path = env_config.get('data_path')
            
            if not data_path:
                raise ValueError(f"Không tìm thấy data_path trong cấu hình môi trường cho mô hình '{model_id}'")
            
            env = TradingEnv(
                data_path=data_path,
                symbol=env_config.get('symbol', 'BTC/USDT'),
                timeframe=env_config.get('timeframe', '1h'),
                initial_balance=env_config.get('initial_balance', 10000.0),
                window_size=env_config.get('window_size', 100),
                random_start=env_config.get('random_start', True),
                reward_function=env_config.get('reward_function', 'profit'),
                fee_rate=env_config.get('fee_rate', 0.001),
                logger=self.logger
            )
        else:
            raise ValueError(f"Loại môi trường không được hỗ trợ: {env_type}")
        
        return env
    
    def _create_evaluation_environment(
        self, 
        model_id: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> BaseEnvironment:
        """
        Tạo môi trường đánh giá.
        
        Args:
            model_id: ID mô hình
            config: Cấu hình đánh giá
            
        Returns:
            BaseEnvironment: Môi trường đánh giá
        """
        # Nếu không có config, sử dụng cấu hình từ model_status
        if config is None and model_id in self.model_status:
            metadata_path = None
            
            # Tìm đường dẫn metadata
            current_version = self.model_status[model_id].get('current_version', 1)
            version_dir = self.models_dir / model_id / f"v{current_version}"
            
            if version_dir.exists():
                metadata_path = version_dir / "metadata.json"
            
            if metadata_path and metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    config = metadata.get('training_config', {})
        
        # Tạo môi trường huấn luyện
        return self._create_training_environment(model_id, config or {})
    
    def _create_agent(
        self, 
        model_type: str, 
        env: BaseEnvironment, 
        config: Dict[str, Any]
    ) -> BaseAgent:
        """
        Tạo agent mới.
        
        Args:
            model_type: Loại mô hình
            env: Môi trường huấn luyện
            config: Cấu hình huấn luyện
            
        Returns:
            BaseAgent: Agent mới
        """
        # Lấy kích thước không gian trạng thái và hành động
        state_dim = env.observation_space.shape
        action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
        
        # Tạo agent dựa trên loại
        if model_type.lower() == 'dqn':
            agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                env=env,
                learning_rate=config.get('learning_rate', 0.001),
                gamma=config.get('gamma', 0.99),
                epsilon=config.get('epsilon', 1.0),
                epsilon_min=config.get('epsilon_min', 0.01),
                epsilon_decay=config.get('epsilon_decay', 0.995),
                batch_size=config.get('batch_size', 64),
                memory_size=config.get('memory_size', 10000),
                update_target_freq=config.get('update_target_freq', 100),
                hidden_layers=config.get('hidden_layers', [64, 64]),
                activation=config.get('activation', 'relu'),
                double_dqn=config.get('double_dqn', False),
                dueling=config.get('dueling', False),
                prioritized_replay=config.get('prioritized_replay', False),
                logger=self.logger
            )
        else:
            raise ValueError(f"Loại agent không được hỗ trợ: {model_type}")
        
        return agent
    
    def _load_model(
        self, 
        model_path: Path, 
        model_info: Dict[str, Any]
    ) -> BaseAgent:
        """
        Tải mô hình từ đường dẫn.
        
        Args:
            model_path: Đường dẫn mô hình
            model_info: Thông tin mô hình
            
        Returns:
            BaseAgent: Mô hình đã tải
        """
        model_type = model_info.get('type', 'dqn')
        
        # Tải metadata nếu có
        metadata = self._load_model_metadata(model_path)
        
        # Tạo môi trường giả
        env_config = metadata.get('training_config', {}).get('environment', {})
        env = self._create_dummy_environment(env_config)
        
        # Tạo agent
        if model_type.lower() == 'dqn':
            # Lấy kích thước không gian trạng thái và hành động
            state_dim = env.observation_space.shape
            action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
            
            # Tạo agent với thông số từ metadata
            agent_config = metadata.get('training_config', {})
            
            agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                env=env,
                learning_rate=agent_config.get('learning_rate', 0.001),
                gamma=agent_config.get('gamma', 0.99),
                epsilon=agent_config.get('epsilon_min', 0.01),  # Sử dụng epsilon_min cho inference
                epsilon_min=agent_config.get('epsilon_min', 0.01),
                epsilon_decay=agent_config.get('epsilon_decay', 0.995),
                batch_size=agent_config.get('batch_size', 64),
                memory_size=agent_config.get('memory_size', 10000),
                update_target_freq=agent_config.get('update_target_freq', 100),
                hidden_layers=agent_config.get('hidden_layers', [64, 64]),
                activation=agent_config.get('activation', 'relu'),
                double_dqn=agent_config.get('double_dqn', False),
                dueling=agent_config.get('dueling', False),
                prioritized_replay=agent_config.get('prioritized_replay', False),
                logger=self.logger
            )
            
            # Tải trọng số
            if model_path.is_dir():
                # Tìm file mô hình
                model_files = list(model_path.glob("*_model"))
                
                if not model_files:
                    model_files = list(model_path.glob("*model*"))
                
                if model_files:
                    model_file = model_files[0]
                    agent._load_model_impl(model_file)
                    self.logger.info(f"Đã tải mô hình từ {model_file}")
                else:
                    raise FileNotFoundError(f"Không tìm thấy file mô hình trong {model_path}")
            else:
                agent._load_model_impl(model_path)
                self.logger.info(f"Đã tải mô hình từ {model_path}")
            
            return agent
        else:
            raise ValueError(f"Loại agent không được hỗ trợ: {model_type}")
    
    def _load_model_metadata(self, model_path: Path) -> Dict[str, Any]:
        """
        Tải metadata của mô hình.
        
        Args:
            model_path: Đường dẫn mô hình
            
        Returns:
            Dict: Metadata mô hình
        """
        # Nếu model_path là thư mục, tìm file metadata
        if model_path.is_dir():
            metadata_path = model_path / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            
            # Không tìm thấy metadata
            return {}
        
        # Nếu model_path là file, tìm metadata trong thư mục cha
        metadata_path = model_path.parent / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        
        # Không tìm thấy metadata
        return {}
    
    def _create_dummy_environment(self, env_config: Dict[str, Any]) -> BaseEnvironment:
        """
        Tạo môi trường giả để tải mô hình.
        
        Args:
            env_config: Cấu hình môi trường
            
        Returns:
            BaseEnvironment: Môi trường giả
        """
        env_type = env_config.get('type', 'trading')
        
        # Tạo môi trường dựa trên loại
        if env_type == 'trading':
            # Tạo dữ liệu giả
            dummy_data = pd.DataFrame({
                'timestamp': pd.date_range(start='2023-01-01', periods=1000, freq='H'),
                'open': np.random.random(1000) * 1000 + 10000,
                'high': np.random.random(1000) * 1000 + 10000,
                'low': np.random.random(1000) * 1000 + 10000,
                'close': np.random.random(1000) * 1000 + 10000,
                'volume': np.random.random(1000) * 100
            })
            
            # Tạo môi trường với dữ liệu giả
            env = TradingEnv(
                data=dummy_data,
                symbol=env_config.get('symbol', 'BTC/USDT'),
                timeframe=env_config.get('timeframe', '1h'),
                window_size=env_config.get('window_size', 100),
                logger=self.logger
            )
            
            return env
        else:
            raise ValueError(f"Loại môi trường không được hỗ trợ: {env_type}")
    
    def _evaluate_model(
        self, 
        model: BaseAgent, 
        env: BaseEnvironment, 
        model_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Đánh giá hiệu suất mô hình.
        
        Args:
            model: Mô hình cần đánh giá
            env: Môi trường đánh giá
            model_info: Thông tin mô hình
            
        Returns:
            Dict: Các chỉ số hiệu suất
        """
        # Cài đặt epsilon thấp để đánh giá
        if hasattr(model, 'epsilon'):
            original_epsilon = model.epsilon
            model.epsilon = model.epsilon_min
        
        try:
            # Thực hiện đánh giá
            eval_episodes = 10
            rewards = []
            episode_lengths = []
            portfolio_values = []
            
            for _ in range(eval_episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
                
                while not done:
                    action = model.act(state, explore=False)
                    next_state, reward, done, info = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    state = next_state
                
                rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Lưu giá trị danh mục nếu có
                if hasattr(env, 'current_nav'):
                    portfolio_values.append(env.current_nav)
            
            # Tính các chỉ số hiệu suất
            metrics = {
                'avg_reward': np.mean(rewards),
                'avg_episode_length': np.mean(episode_lengths),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards),
                'std_reward': np.std(rewards)
            }
            
            # Tính các chỉ số tài chính nếu có giá trị danh mục
            if portfolio_values:
                # Tạo chuỗi equity cho PerformanceMetrics
                equity_series = pd.Series(
                    portfolio_values, 
                    index=pd.date_range(start='2023-01-01', periods=len(portfolio_values), freq='D')
                )
                
                # Tính toán các chỉ số hiệu suất tài chính
                perf_metrics = PerformanceMetrics(equity_series)
                financial_metrics = perf_metrics.calculate_all_metrics()
                
                # Thêm vào metrics
                metrics['total_return'] = financial_metrics.get('total_return', 0.0)
                metrics['sharpe_ratio'] = financial_metrics.get('sharpe_ratio', 0.0)
                metrics['max_drawdown'] = financial_metrics.get('max_drawdown', 0.0)
                metrics['calmar_ratio'] = financial_metrics.get('calmar_ratio', 0.0)
                metrics['volatility'] = financial_metrics.get('volatility', 0.0)
                metrics['sortino_ratio'] = financial_metrics.get('sortino_ratio', 0.0)
            
            self.logger.info(f"Kết quả đánh giá mô hình: {metrics}")
            
            return metrics
            
        finally:
            # Khôi phục epsilon
            if hasattr(model, 'epsilon'):
                model.epsilon = original_epsilon
    
    def _save_model_status(self) -> bool:
        """
        Lưu trạng thái mô hình vào file.
        
        Returns:
            bool: True nếu lưu thành công, False nếu không
        """
        try:
            # Tạo đường dẫn file
            status_path = self.models_dir / "model_status.json"
            
            # Tạo dữ liệu trạng thái
            status_data = {
                'model_status': self.model_status,
                'retraining_history': self.retraining_history,
                'timestamp': datetime.now().isoformat()
            }
            
            # Lưu file
            with open(status_path, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, indent=4, ensure_ascii=False, default=str)
            
            self.logger.debug(f"Đã lưu trạng thái mô hình tại {status_path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu trạng thái mô hình: {str(e)}")
            return False
    
    def _load_model_status(self) -> bool:
        """
        Tải trạng thái mô hình từ file.
        
        Returns:
            bool: True nếu tải thành công, False nếu không
        """
        status_path = self.models_dir / "model_status.json"
        
        if not status_path.exists():
            self.logger.info("Không tìm thấy file trạng thái mô hình để tải")
            return False
        
        try:
            # Đọc file
            with open(status_path, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
            
            # Cập nhật trạng thái
            if 'model_status' in status_data:
                self.model_status = status_data['model_status']
            
            if 'retraining_history' in status_data:
                self.retraining_history = status_data['retraining_history']
            
            self.logger.info(f"Đã tải trạng thái mô hình từ {status_path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tải trạng thái mô hình: {str(e)}")
            return False