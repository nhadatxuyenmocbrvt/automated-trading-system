"""
Engine suy luận thời gian thực.
File này cung cấp lớp InferenceEngine để thực hiện suy luận từ các mô hình đã huấn luyện
trong thời gian thực, xử lý dữ liệu trực tuyến, và tạo tín hiệu giao dịch.
"""

import os
import time
import json
import signal
import logging
import threading
import traceback
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
import concurrent.futures
from collections import deque

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config, MODEL_DIR
from config.constants import OrderType, PositionSide
from data_processors.data_pipeline import DataPipeline
from real_time_inference.auto_restart.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
from real_time_inference.system_monitor.notification_manager import NotificationManager, NotificationPriority, NotificationType
from real_time_inference.auto_restart.recovery_system import RecoverySystem, RecoveryAction

class ModelType(Enum):
    """Loại mô hình."""
    DQN = "dqn"
    PPO = "ppo"
    A2C = "a2c"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"

class SignalType(Enum):
    """Loại tín hiệu."""
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    HOLD = "hold"

class InferenceMode(Enum):
    """Chế độ suy luận."""
    SINGLE_MODEL = "single_model"      # Sử dụng một mô hình duy nhất
    ENSEMBLE = "ensemble"              # Sử dụng kết hợp nhiều mô hình
    VOTING = "voting"                  # Sử dụng bỏ phiếu từ nhiều mô hình
    SEQUENTIAL = "sequential"          # Sử dụng nhiều mô hình theo trình tự
    HIERARCHICAL = "hierarchical"      # Sử dụng mô hình phân cấp

class ModelConfig:
    """
    Lớp cấu hình mô hình.
    Lưu trữ các thông tin cấu hình và tham số cho một mô hình cụ thể.
    """
    
    def __init__(
        self,
        model_id: str,
        model_type: ModelType,
        model_path: str,
        symbol: str,
        timeframe: str,
        input_features: List[str],
        input_window: int,
        risk_threshold: float = 0.5,
        confidence_threshold: float = 0.6,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Khởi tạo cấu hình mô hình.
        
        Args:
            model_id: ID mô hình
            model_type: Loại mô hình
            model_path: Đường dẫn file mô hình
            symbol: Cặp giao dịch
            timeframe: Khung thời gian
            input_features: Danh sách đặc trưng đầu vào
            input_window: Kích thước cửa sổ dữ liệu đầu vào
            risk_threshold: Ngưỡng rủi ro
            confidence_threshold: Ngưỡng độ tin cậy
            parameters: Các tham số bổ sung
        """
        self.model_id = model_id
        self.model_type = model_type
        self.model_path = model_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.input_features = input_features
        self.input_window = input_window
        self.risk_threshold = risk_threshold
        self.confidence_threshold = confidence_threshold
        self.parameters = parameters or {}
        
        # Thời gian tạo
        self.created_at = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi cấu hình thành dict.
        
        Returns:
            Dict chứa thông tin cấu hình
        """
        return {
            'model_id': self.model_id,
            'model_type': self.model_type.value if isinstance(self.model_type, ModelType) else self.model_type,
            'model_path': self.model_path,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'input_features': self.input_features,
            'input_window': self.input_window,
            'risk_threshold': self.risk_threshold,
            'confidence_threshold': self.confidence_threshold,
            'parameters': self.parameters,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """
        Tạo cấu hình từ dict.
        
        Args:
            data: Dict chứa thông tin cấu hình
            
        Returns:
            Đối tượng ModelConfig
        """
        # Chuyển đổi model_type từ string sang Enum
        model_type = data.get('model_type')
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type)
            except ValueError:
                # Nếu không chuyển đổi được, giữ nguyên
                pass
        
        # Tạo đối tượng
        config = cls(
            model_id=data.get('model_id'),
            model_type=model_type,
            model_path=data.get('model_path'),
            symbol=data.get('symbol'),
            timeframe=data.get('timeframe'),
            input_features=data.get('input_features', []),
            input_window=data.get('input_window'),
            risk_threshold=data.get('risk_threshold', 0.5),
            confidence_threshold=data.get('confidence_threshold', 0.6),
            parameters=data.get('parameters', {})
        )
        
        # Thêm created_at nếu có
        if 'created_at' in data:
            config.created_at = data['created_at']
            
        return config

class InferenceRequest:
    """
    Lớp yêu cầu suy luận.
    Đại diện cho một yêu cầu suy luận được gửi đến engine.
    """
    
    def __init__(
        self,
        request_id: str,
        model_id: str,
        data: pd.DataFrame,
        timestamp: Optional[datetime] = None,
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 1,
        callback: Optional[Callable] = None
    ):
        """
        Khởi tạo yêu cầu suy luận.
        
        Args:
            request_id: ID yêu cầu
            model_id: ID mô hình
            data: DataFrame dữ liệu đầu vào
            timestamp: Thời gian yêu cầu
            parameters: Các tham số bổ sung
            priority: Độ ưu tiên (1-10, càng cao càng ưu tiên)
            callback: Hàm callback khi hoàn thành
        """
        self.request_id = request_id
        self.model_id = model_id
        self.data = data
        self.timestamp = timestamp or datetime.now()
        self.parameters = parameters or {}
        self.priority = priority
        self.callback = callback
        
        # Thời gian tạo và xử lý
        self.created_at = datetime.now()
        self.processed_at = None
        self.processing_time = None
        
    def __lt__(self, other: 'InferenceRequest') -> bool:
        """So sánh ưu tiên để sử dụng trong PriorityQueue."""
        # Ưu tiên cao hơn sẽ được xử lý trước
        return self.priority > other.priority

class InferenceResult:
    """
    Lớp kết quả suy luận.
    Đại diện cho kết quả suy luận từ một mô hình.
    """
    
    def __init__(
        self,
        request_id: str,
        model_id: str,
        signal: SignalType,
        confidence: float,
        timestamp: datetime,
        position_size: Optional[float] = None,
        price: Optional[float] = None,
        duration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Khởi tạo kết quả suy luận.
        
        Args:
            request_id: ID yêu cầu
            model_id: ID mô hình
            signal: Loại tín hiệu
            confidence: Độ tin cậy của tín hiệu (0.0-1.0)
            timestamp: Thời gian tạo kết quả
            position_size: Kích thước vị thế đề xuất
            price: Giá đề xuất
            duration: Thời gian dự kiến giữ vị thế
            metadata: Metadata bổ sung
        """
        self.request_id = request_id
        self.model_id = model_id
        self.signal = signal
        self.confidence = confidence
        self.timestamp = timestamp
        self.position_size = position_size
        self.price = price
        self.duration = duration
        self.metadata = metadata or {}
        
        # Thời gian tạo
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi kết quả thành dict.
        
        Returns:
            Dict chứa thông tin kết quả
        """
        return {
            'request_id': self.request_id,
            'model_id': self.model_id,
            'signal': self.signal.value if isinstance(self.signal, SignalType) else self.signal,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'position_size': self.position_size,
            'price': self.price,
            'duration': self.duration,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }

class InferenceEngine:
    """
    Engine suy luận thời gian thực.
    Cung cấp cơ chế để thực hiện suy luận từ các mô hình đã được huấn luyện
    trong thời gian thực, xử lý dữ liệu trực tuyến và tạo ra các tín hiệu giao dịch.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_dir: Optional[str] = None,
        data_buffer_size: int = 1000,
        worker_threads: int = 4,
        inference_mode: InferenceMode = InferenceMode.SINGLE_MODEL,
        error_handler: Optional[ErrorHandler] = None,
        notification_manager: Optional[NotificationManager] = None,
        recovery_system: Optional[RecoverySystem] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo InferenceEngine.
        
        Args:
            config_path: Đường dẫn file cấu hình
            model_dir: Thư mục chứa mô hình
            data_buffer_size: Kích thước buffer dữ liệu
            worker_threads: Số luồng worker
            inference_mode: Chế độ suy luận
            error_handler: Xử lý lỗi
            notification_manager: Quản lý thông báo
            recovery_system: Hệ thống phục hồi
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("inference_engine")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config().get_all()
        
        # Thiết lập thư mục chứa mô hình
        if model_dir is None:
            self.model_dir = MODEL_DIR
        else:
            self.model_dir = Path(model_dir)
        
        # Đảm bảo thư mục tồn tại
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        
        # Lưu trữ thông tin cấu hình
        self.config_path = config_path
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()
        
        # Thiết lập các tham số
        self.data_buffer_size = data_buffer_size
        self.worker_threads = worker_threads
        self.inference_mode = inference_mode
        
        # Xử lý lỗi
        if error_handler is None:
            self.error_handler = ErrorHandler(
                logger=self.logger.getChild("error_handler")
            )
        else:
            self.error_handler = error_handler
        
        # Quản lý thông báo
        self.notification_manager = notification_manager
        
        # Hệ thống phục hồi
        self.recovery_system = recovery_system
        
        # Khởi tạo DataPipeline
        self.data_pipeline = DataPipeline(
            logger=self.logger.getChild("data_pipeline")
        )
        
        # Khởi tạo buffer dữ liệu
        self.data_buffers = {}  # {symbol: deque(DataFrame)}
        
        # Khởi tạo danh sách mô hình
        self.models = {}
        self.model_configs = {}
        
        # Tạo queue yêu cầu
        self.request_queue = Queue()
        self.results_buffer = deque(maxlen=1000)
        
        # Khởi tạo trạng thái
        self.is_running = False
        self.worker_pool = None
        self.worker_threads_list = []
        
        # Khởi tạo lock thread
        self.model_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.result_lock = threading.Lock()
        
        # Khởi tạo thông số theo dõi
        self.stats = {
            'processed_requests': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'avg_processing_time': 0,
            'signals_generated': {
                'buy': 0,
                'sell': 0,
                'close': 0,
                'hold': 0
            },
            'models_usage': {}
        }
        
        # Theo dõi hiệu suất
        self.performance_metrics = {
            'processing_times': [],
            'requests_per_minute': 0,
            'load_average': 0.0,
            'error_rate': 0.0
        }
        
        self.logger.info(f"Đã khởi tạo InferenceEngine với chế độ {inference_mode.value}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Lấy cấu hình mặc định.
        
        Returns:
            Dict chứa cấu hình
        """
        return {
            "engine_config": {
                "data_buffer_size": 1000,
                "worker_threads": 4,
                "inference_mode": "single_model",
                "batch_size": 32,
                "enable_logging": True,
                "enable_profiling": False,
                "health_check_interval": 60
            },
            "model_configs": {
                "dqn_btc_1h": {
                    "model_id": "dqn_btc_1h",
                    "model_type": "dqn",
                    "model_path": "models/dqn/BTC_USDT/1h/best_model",
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "input_features": ["open", "high", "low", "close", "volume"],
                    "input_window": 100,
                    "risk_threshold": 0.5,
                    "confidence_threshold": 0.6,
                    "parameters": {
                        "state_dim": [100, 5],
                        "action_dim": 3
                    }
                }
            },
            "signal_config": {
                "enable_autosizing": True,
                "default_position_size": 0.1,
                "max_position_size": 0.5,
                "risk_per_trade": 0.02,
                "signal_expiry": 300,
                "apply_filters": True,
                "filter_settings": {
                    "min_confidence": 0.6,
                    "trend_filter": True,
                    "volatility_filter": True
                }
            },
            "feature_config": {
                "technical_indicators": {
                    "use_ta": True,
                    "indicators": ["sma", "ema", "rsi", "macd", "bbands", "atr"]
                },
                "normalization": {
                    "method": "min_max",
                    "window": 100
                },
                "feature_engineering": {
                    "pca": False,
                    "wavelets": False,
                    "fourier": False
                }
            }
        }
    
    def start(self) -> bool:
        """
        Khởi động engine.
        
        Returns:
            True nếu thành công, False nếu không
        """
        if self.is_running:
            self.logger.warning("InferenceEngine đã đang chạy")
            return True
        
        try:
            self.logger.info("Khởi động InferenceEngine")
            
            # Tải các mô hình
            self._load_models()
            
            # Khởi tạo pool workers
            self.worker_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.worker_threads,
                thread_name_prefix="inference_worker"
            )
            
            # Khởi tạo thread xử lý
            self.processor_thread = threading.Thread(
                target=self._process_requests,
                daemon=True,
                name="inference_processor"
            )
            
            # Khởi tạo thread giám sát
            self.monitor_thread = threading.Thread(
                target=self._monitor_performance,
                daemon=True,
                name="inference_monitor"
            )
            
            # Thiết lập handler cho các tín hiệu
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Đánh dấu đang chạy
            self.is_running = True
            
            # Khởi động các thread
            self.processor_thread.start()
            self.monitor_thread.start()
            
            self.logger.info("InferenceEngine đã khởi động thành công")
            
            # Gửi thông báo nếu có
            if self.notification_manager:
                self.notification_manager.send_system_notification(
                    component="inference_engine", 
                    status="started", 
                    details="InferenceEngine đã khởi động thành công"
                )
            
            return True
            
        except Exception as e:
            error_msg = f"Lỗi khi khởi động InferenceEngine: {str(e)}"
            error_trace = traceback.format_exc()
            
            self.logger.error(f"{error_msg}\n{error_trace}")
            
            # Xử lý lỗi
            if self.error_handler:
                self.error_handler.handle_error(
                    error=e,
                    component="inference_engine",
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.HIGH,
                    context={"action": "start", "trace": error_trace}
                )
            
            # Gửi thông báo nếu có
            if self.notification_manager:
                self.notification_manager.send_error_notification(
                    error_message=error_msg,
                    error_source="inference_engine",
                    error_trace=error_trace,
                    critical=True
                )
            
            return False
    
    def stop(self) -> bool:
        """
        Dừng engine.
        
        Returns:
            True nếu thành công, False nếu không
        """
        if not self.is_running:
            self.logger.warning("InferenceEngine đã dừng")
            return True
        
        try:
            self.logger.info("Đang dừng InferenceEngine")
            
            # Đánh dấu dừng hoạt động
            self.is_running = False
            
            # Đợi thread xử lý kết thúc
            if self.processor_thread and self.processor_thread.is_alive():
                self.processor_thread.join(timeout=30)
            
            # Đợi thread giám sát kết thúc
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=10)
            
            # Dừng worker pool
            if self.worker_pool:
                self.worker_pool.shutdown(wait=True)
                self.worker_pool = None
            
            # Xóa các thread worker
            self.worker_threads_list = []
            
            # Lưu trạng thái các mô hình nếu cần
            self._save_models_state()
            
            self.logger.info("InferenceEngine đã dừng thành công")
            
            # Gửi thông báo nếu có
            if self.notification_manager:
                self.notification_manager.send_system_notification(
                    component="inference_engine", 
                    status="stopped", 
                    details="InferenceEngine đã dừng thành công"
                )
            
            return True
            
        except Exception as e:
            error_msg = f"Lỗi khi dừng InferenceEngine: {str(e)}"
            error_trace = traceback.format_exc()
            
            self.logger.error(f"{error_msg}\n{error_trace}")
            
            # Xử lý lỗi
            if self.error_handler:
                self.error_handler.handle_error(
                    error=e,
                    component="inference_engine",
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.MEDIUM,
                    context={"action": "stop", "trace": error_trace}
                )
            
            return False
    
    def _signal_handler(self, sig, frame) -> None:
        """
        Xử lý tín hiệu hệ thống.
        
        Args:
            sig: Tín hiệu
            frame: Frame
        """
        self.logger.info(f"Đã nhận tín hiệu {sig}, đang dừng InferenceEngine...")
        self.stop()
    
    def _load_models(self) -> None:
        """
        Tải các mô hình từ cấu hình.
        """
        self.logger.info("Đang tải các mô hình...")
        
        # Lấy cấu hình mô hình
        model_configs = self.config.get("model_configs", {})
        
        # Tải từng mô hình
        for model_id, model_config_data in model_configs.items():
            try:
                # Tạo cấu hình mô hình
                model_config = ModelConfig.from_dict(model_config_data)
                
                # Tải mô hình
                model = self._load_model(model_config)
                
                if model:
                    with self.model_lock:
                        self.models[model_id] = model
                        self.model_configs[model_id] = model_config
                        
                    self.logger.info(f"Đã tải mô hình {model_id} thành công")
                    
                    # Cập nhật stats
                    self.stats['models_usage'][model_id] = 0
                else:
                    self.logger.error(f"Không thể tải mô hình {model_id}")
                    
            except Exception as e:
                error_msg = f"Lỗi khi tải mô hình {model_id}: {str(e)}"
                error_trace = traceback.format_exc()
                
                self.logger.error(f"{error_msg}\n{error_trace}")
                
                # Xử lý lỗi
                if self.error_handler:
                    self.error_handler.handle_error(
                        error=e,
                        component="inference_engine",
                        category=ErrorCategory.MODEL,
                        severity=ErrorSeverity.MEDIUM,
                        context={"model_id": model_id, "action": "load", "trace": error_trace}
                    )
        
        self.logger.info(f"Đã tải {len(self.models)} mô hình")
    
    def _load_model(self, model_config: ModelConfig) -> Any:
        """
        Tải một mô hình từ file.
        
        Args:
            model_config: Cấu hình mô hình
            
        Returns:
            Đối tượng mô hình đã tải
        """
        model_type = model_config.model_type
        model_path = model_config.model_path
        model_id = model_config.model_id
        parameters = model_config.parameters
        
        # Kiểm tra đường dẫn mô hình
        if not os.path.exists(model_path):
            full_path = os.path.join(self.model_dir, model_path)
            if os.path.exists(full_path):
                model_path = full_path
            else:
                self.logger.error(f"Không tìm thấy file mô hình: {model_path}")
                return None
        
        # Tải mô hình dựa trên loại
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type)
            except ValueError:
                model_type = ModelType.CUSTOM
        
        try:
            if model_type == ModelType.DQN:
                from models.agents.dqn_agent import DQNAgent
                
                # Lấy thông tin state_dim và action_dim
                state_dim = parameters.get('state_dim')
                action_dim = parameters.get('action_dim')
                
                if not state_dim or not action_dim:
                    self.logger.error(f"Thiếu thông tin state_dim hoặc action_dim cho mô hình {model_id}")
                    return None
                
                # Tạo agent
                agent = DQNAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    **{k: v for k, v in parameters.items() if k not in ['state_dim', 'action_dim']}
                )
                
                # Tải mô hình
                if agent.load_model(model_path):
                    return agent
                else:
                    self.logger.error(f"Không thể tải mô hình DQN từ {model_path}")
                    return None
                
            elif model_type == ModelType.PPO:
                from models.agents.ppo_agent import PPOAgent
                
                # Lấy thông tin state_dim và action_dim
                state_dim = parameters.get('state_dim')
                action_dim = parameters.get('action_dim')
                
                if not state_dim or not action_dim:
                    self.logger.error(f"Thiếu thông tin state_dim hoặc action_dim cho mô hình {model_id}")
                    return None
                
                # Tạo agent
                agent = PPOAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    **{k: v for k, v in parameters.items() if k not in ['state_dim', 'action_dim']}
                )
                
                # Tải mô hình
                if agent.load_model(model_path):
                    return agent
                else:
                    self.logger.error(f"Không thể tải mô hình PPO từ {model_path}")
                    return None
                
            elif model_type == ModelType.A2C:
                from models.agents.a2c_agent import A2CAgent
                
                # Lấy thông tin state_dim và action_dim
                state_dim = parameters.get('state_dim')
                action_dim = parameters.get('action_dim')
                
                if not state_dim or not action_dim:
                    self.logger.error(f"Thiếu thông tin state_dim hoặc action_dim cho mô hình {model_id}")
                    return None
                
                # Tạo agent
                agent = A2CAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    **{k: v for k, v in parameters.items() if k not in ['state_dim', 'action_dim']}
                )
                
                # Tải mô hình
                if agent.load_model(model_path):
                    return agent
                else:
                    self.logger.error(f"Không thể tải mô hình A2C từ {model_path}")
                    return None
                
            elif model_type == ModelType.ENSEMBLE:
                # Tải các mô hình con
                sub_models = parameters.get('sub_models', [])
                
                if not sub_models:
                    self.logger.error(f"Không có sub_models cho mô hình ensemble {model_id}")
                    return None
                
                # Tạo và tải các mô hình con
                ensemble = {}
                for sub_model_config in sub_models:
                    sub_model_id = sub_model_config.get('model_id')
                    sub_model = self._load_model(ModelConfig.from_dict(sub_model_config))
                    if sub_model:
                        ensemble[sub_model_id] = sub_model
                    else:
                        self.logger.warning(f"Không thể tải mô hình con {sub_model_id} cho ensemble {model_id}")
                
                if ensemble:
                    return {
                        'type': 'ensemble',
                        'models': ensemble,
                        'weights': parameters.get('weights', {}),
                        'method': parameters.get('method', 'average')
                    }
                else:
                    self.logger.error(f"Không có mô hình con nào được tải cho ensemble {model_id}")
                    return None
                
            elif model_type == ModelType.CUSTOM:
                # Tải mô hình tùy chỉnh
                custom_loader = parameters.get('custom_loader')
                
                if not custom_loader:
                    self.logger.error(f"Không có custom_loader cho mô hình {model_id}")
                    return None
                
                try:
                    # Tải mô hình sử dụng custom_loader
                    module_path, function_name = custom_loader.rsplit('.', 1)
                    module = __import__(module_path, fromlist=[function_name])
                    loader_function = getattr(module, function_name)
                    
                    return loader_function(model_path, parameters)
                    
                except Exception as e:
                    self.logger.error(f"Lỗi khi tải mô hình tùy chỉnh {model_id}: {str(e)}")
                    return None
            else:
                self.logger.error(f"Loại mô hình không được hỗ trợ: {model_type}")
                return None
                
        except ImportError as e:
            self.logger.error(f"Không thể import module cần thiết cho mô hình {model_id}: {str(e)}")
            return None
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải mô hình {model_id}: {str(e)}")
            return None
    
    def _save_models_state(self) -> None:
        """
        Lưu trạng thái của các mô hình.
        """
        self.logger.info("Đang lưu trạng thái các mô hình...")
        
        # Chưa triển khai đầy đủ - phụ thuộc vào loại mô hình cụ thể
        pass
    
    def _process_requests(self) -> None:
        """
        Xử lý các yêu cầu từ queue.
        Chạy trong một thread riêng biệt.
        """
        self.logger.info("Thread xử lý yêu cầu đã khởi động")
        
        while self.is_running:
            try:
                # Lấy yêu cầu từ queue (không chờ nếu không có)
                try:
                    request = self.request_queue.get(block=True, timeout=1.0)
                except Empty:
                    continue
                
                # Kiểm tra xem yêu cầu có hợp lệ không
                if not isinstance(request, InferenceRequest):
                    self.logger.warning(f"Loại yêu cầu không hợp lệ: {type(request)}")
                    self.request_queue.task_done()
                    continue
                
                # Kiểm tra xem model_id có tồn tại không
                model_id = request.model_id
                if model_id not in self.models:
                    self.logger.warning(f"Mô hình {model_id} không tồn tại")
                    
                    # Gọi callback với kết quả thất bại
                    if request.callback:
                        error_result = InferenceResult(
                            request_id=request.request_id,
                            model_id=model_id,
                            signal=SignalType.HOLD,
                            confidence=0.0,
                            timestamp=datetime.now(),
                            metadata={'error': f"Mô hình {model_id} không tồn tại"}
                        )
                        request.callback(error_result)
                    
                    self.request_queue.task_done()
                    continue
                
                # Ghi log bắt đầu xử lý
                self.logger.debug(f"Đang xử lý yêu cầu {request.request_id} cho mô hình {model_id}")
                
                # Đánh dấu thời gian bắt đầu xử lý
                start_time = time.time()
                request.processed_at = datetime.now()
                
                # Thực hiện suy luận
                if self.inference_mode == InferenceMode.SINGLE_MODEL:
                    # Gửi yêu cầu đến worker pool
                    future = self.worker_pool.submit(
                        self._perform_inference,
                        request
                    )
                    future.add_done_callback(lambda f: self._handle_inference_result(f, request))
                    
                elif self.inference_mode == InferenceMode.ENSEMBLE:
                    # Gửi yêu cầu đến nhiều mô hình và tổng hợp kết quả
                    future = self.worker_pool.submit(
                        self._perform_ensemble_inference,
                        request
                    )
                    future.add_done_callback(lambda f: self._handle_inference_result(f, request))
                    
                elif self.inference_mode == InferenceMode.VOTING:
                    # Gửi yêu cầu đến nhiều mô hình và bỏ phiếu
                    future = self.worker_pool.submit(
                        self._perform_voting_inference,
                        request
                    )
                    future.add_done_callback(lambda f: self._handle_inference_result(f, request))
                    
                elif self.inference_mode == InferenceMode.SEQUENTIAL:
                    # Gửi yêu cầu đến các mô hình theo trình tự
                    future = self.worker_pool.submit(
                        self._perform_sequential_inference,
                        request
                    )
                    future.add_done_callback(lambda f: self._handle_inference_result(f, request))
                    
                elif self.inference_mode == InferenceMode.HIERARCHICAL:
                    # Gửi yêu cầu đến các mô hình phân cấp
                    future = self.worker_pool.submit(
                        self._perform_hierarchical_inference,
                        request
                    )
                    future.add_done_callback(lambda f: self._handle_inference_result(f, request))
                
                else:
                    self.logger.error(f"Chế độ suy luận không được hỗ trợ: {self.inference_mode}")
                    
                    # Gọi callback với kết quả thất bại
                    if request.callback:
                        error_result = InferenceResult(
                            request_id=request.request_id,
                            model_id=model_id,
                            signal=SignalType.HOLD,
                            confidence=0.0,
                            timestamp=datetime.now(),
                            metadata={'error': f"Chế độ suy luận không được hỗ trợ: {self.inference_mode}"}
                        )
                        request.callback(error_result)
                    
                    # Cập nhật thống kê
                    self.stats['failed_inferences'] += 1
                
                # Đánh dấu task đã hoàn thành trong queue
                self.request_queue.task_done()
                
                # Tính thời gian xử lý
                request.processing_time = time.time() - start_time
                
                # Cập nhật thống kê
                self.stats['processed_requests'] += 1
                
                # Cập nhật thời gian xử lý trung bình
                total_time = self.stats['avg_processing_time'] * (self.stats['processed_requests'] - 1) + request.processing_time
                self.stats['avg_processing_time'] = total_time / self.stats['processed_requests'] if self.stats['processed_requests'] > 0 else 0
                
                # Lưu thời gian xử lý để đánh giá hiệu suất
                self.performance_metrics['processing_times'].append(request.processing_time)
                # Giới hạn số lượng thời gian để tiết kiệm bộ nhớ
                if len(self.performance_metrics['processing_times']) > 1000:
                    self.performance_metrics['processing_times'] = self.performance_metrics['processing_times'][-1000:]
                
            except Exception as e:
                error_msg = f"Lỗi trong thread xử lý yêu cầu: {str(e)}"
                error_trace = traceback.format_exc()
                
                self.logger.error(f"{error_msg}\n{error_trace}")
                
                # Xử lý lỗi
                if self.error_handler:
                    self.error_handler.handle_error(
                        error=e,
                        component="inference_engine.processor",
                        category=ErrorCategory.SYSTEM,
                        severity=ErrorSeverity.HIGH,
                        context={"action": "process_requests", "trace": error_trace},
                        retry_action=self._restart_processor_thread
                    )
        
        self.logger.info("Thread xử lý yêu cầu đã kết thúc")
    
    def _restart_processor_thread(self) -> bool:
        """
        Khởi động lại thread xử lý.
        
        Returns:
            True nếu thành công, False nếu không
        """
        if not self.is_running:
            self.logger.warning("Không thể khởi động lại thread xử lý khi InferenceEngine không chạy")
            return False
        
        try:
            # Kiểm tra thread hiện tại
            if self.processor_thread and self.processor_thread.is_alive():
                self.logger.warning("Thread xử lý vẫn đang chạy, không cần khởi động lại")
                return True
            
            # Tạo thread mới
            self.processor_thread = threading.Thread(
                target=self._process_requests,
                daemon=True,
                name="inference_processor"
            )
            
            # Khởi động thread
            self.processor_thread.start()
            
            self.logger.info("Đã khởi động lại thread xử lý thành công")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi động lại thread xử lý: {str(e)}")
            return False
    
    def _monitor_performance(self) -> None:
        """
        Giám sát hiệu suất của engine.
        Chạy trong một thread riêng biệt.
        """
        self.logger.info("Thread giám sát hiệu suất đã khởi động")
        
        last_check_time = time.time()
        last_request_count = 0
        
        while self.is_running:
            try:
                # Đợi 1 phút
                time.sleep(60)
                
                # Tính số yêu cầu mỗi phút
                current_time = time.time()
                time_diff = current_time - last_check_time
                
                current_request_count = self.stats['processed_requests']
                request_diff = current_request_count - last_request_count
                
                if time_diff > 0:
                    requests_per_minute = request_diff / (time_diff / 60)
                    self.performance_metrics['requests_per_minute'] = requests_per_minute
                
                # Cập nhật giá trị cuối
                last_check_time = current_time
                last_request_count = current_request_count
                
                # Tính tỷ lệ lỗi
                total_inferences = self.stats['successful_inferences'] + self.stats['failed_inferences']
                if total_inferences > 0:
                    error_rate = self.stats['failed_inferences'] / total_inferences
                    self.performance_metrics['error_rate'] = error_rate
                
                # Kiểm tra tải CPU
                try:
                    import psutil
                    self.performance_metrics['load_average'] = psutil.cpu_percent() / 100.0
                except ImportError:
                    pass
                
                # Ghi log hiệu suất
                self.logger.info(
                    f"Hiệu suất: {self.performance_metrics['requests_per_minute']:.2f} yêu cầu/phút, "
                    f"thời gian xử lý: {self.stats['avg_processing_time']*1000:.2f}ms, "
                    f"tỷ lệ lỗi: {self.performance_metrics['error_rate']*100:.2f}%"
                )
                
                # Kiểm tra nếu tỷ lệ lỗi quá cao
                if self.performance_metrics['error_rate'] > 0.3:  # 30%
                    self.logger.warning(f"Tỷ lệ lỗi cao: {self.performance_metrics['error_rate']*100:.2f}%, cần xem xét")
                    
                    # Gửi thông báo nếu có
                    if self.notification_manager:
                        self.notification_manager.send_system_notification(
                            component="inference_engine", 
                            status="warning", 
                            details=f"Tỷ lệ lỗi cao: {self.performance_metrics['error_rate']*100:.2f}%",
                            priority=NotificationPriority.HIGH
                        )
                
            except Exception as e:
                error_msg = f"Lỗi trong thread giám sát hiệu suất: {str(e)}"
                error_trace = traceback.format_exc()
                
                self.logger.error(f"{error_msg}\n{error_trace}")
                
                # Xử lý lỗi
                if self.error_handler:
                    self.error_handler.handle_error(
                        error=e,
                        component="inference_engine.monitor",
                        category=ErrorCategory.SYSTEM,
                        severity=ErrorSeverity.MEDIUM,
                        context={"action": "monitor_performance", "trace": error_trace}
                    )
        
        self.logger.info("Thread giám sát hiệu suất đã kết thúc")
    
    def _perform_inference(self, request: InferenceRequest) -> InferenceResult:
        """
        Thực hiện suy luận cho một yêu cầu.
        
        Args:
            request: Yêu cầu suy luận
            
        Returns:
            Kết quả suy luận
        """
        model_id = request.model_id
        
        try:
            # Lấy mô hình và cấu hình
            with self.model_lock:
                model = self.models.get(model_id)
                model_config = self.model_configs.get(model_id)
            
            if not model or not model_config:
                error_msg = f"Không tìm thấy mô hình {model_id}"
                self.logger.error(error_msg)
                
                return InferenceResult(
                    request_id=request.request_id,
                    model_id=model_id,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': error_msg}
                )
            
            # Chuẩn bị dữ liệu đầu vào
            features_data = self._prepare_features(request.data, model_config)
            
            # Dự đoán hành động
            model_type = model_config.model_type
            
            if isinstance(model_type, str):
                model_type = ModelType(model_type) if model_type in [m.value for m in ModelType] else ModelType.CUSTOM
            
            # Dựa trên loại mô hình
            if model_type == ModelType.DQN:
                action, action_values = model.get_action(features_data, deterministic=True, return_q_values=True)
                
                # Tính độ tin cậy
                confidence = self._calculate_confidence(action_values)
                
                # Chuyển đổi hành động thành tín hiệu
                signal = self._convert_action_to_signal(action, model_config)
                
            elif model_type == ModelType.PPO or model_type == ModelType.A2C:
                action, action_probs = model.get_action(features_data, deterministic=True, return_probs=True)
                
                # Tính độ tin cậy
                confidence = self._calculate_confidence(action_probs)
                
                # Chuyển đổi hành động thành tín hiệu
                signal = self._convert_action_to_signal(action, model_config)
                
            elif model_type == ModelType.ENSEMBLE:
                # Xử lý riêng cho ensemble, đã được triển khai trong _perform_ensemble_inference
                # Không nên chạy vào đây
                error_msg = "Loại mô hình ensemble không nên được xử lý trong _perform_inference"
                self.logger.error(error_msg)
                
                return InferenceResult(
                    request_id=request.request_id,
                    model_id=model_id,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': error_msg}
                )
            
            elif model_type == ModelType.CUSTOM:
                # Với mô hình tùy chỉnh, cần có hàm predict
                if hasattr(model, 'predict'):
                    result = model.predict(features_data)
                    
                    # Parse kết quả
                    if isinstance(result, tuple) and len(result) >= 2:
                        action, confidence = result[0], result[1]
                    else:
                        action, confidence = result, 0.5
                    
                    # Chuyển đổi hành động thành tín hiệu
                    signal = self._convert_action_to_signal(action, model_config)
                    
                else:
                    error_msg = f"Mô hình tùy chỉnh {model_id} không có hàm predict"
                    self.logger.error(error_msg)
                    
                    return InferenceResult(
                        request_id=request.request_id,
                        model_id=model_id,
                        signal=SignalType.HOLD,
                        confidence=0.0,
                        timestamp=datetime.now(),
                        metadata={'error': error_msg}
                    )
            
            else:
                error_msg = f"Loại mô hình không được hỗ trợ: {model_type}"
                self.logger.error(error_msg)
                
                return InferenceResult(
                    request_id=request.request_id,
                    model_id=model_id,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': error_msg}
                )
            
            # Kiểm tra ngưỡng tin cậy
            confidence_threshold = model_config.confidence_threshold
            
            if confidence < confidence_threshold:
                # Nếu độ tin cậy thấp, chuyển sang tín hiệu HOLD
                signal = SignalType.HOLD
                
                self.logger.debug(f"Độ tin cậy ({confidence:.4f}) dưới ngưỡng ({confidence_threshold}), chuyển sang HOLD")
            
            # Tính kích thước vị thế nếu cần
            position_size = None
            if signal in [SignalType.BUY, SignalType.SELL]:
                # Lấy cấu hình tín hiệu
                signal_config = self.config.get("signal_config", {})
                
                if signal_config.get("enable_autosizing", True):
                    position_size = self._calculate_position_size(
                        confidence=confidence,
                        signal=signal,
                        data=request.data,
                        risk_per_trade=signal_config.get("risk_per_trade", 0.02),
                        max_position_size=signal_config.get("max_position_size", 0.5)
                    )
                else:
                    position_size = signal_config.get("default_position_size", 0.1)
            
            # Áp dụng bộ lọc nếu cần
            if self.config.get("signal_config", {}).get("apply_filters", True):
                signal = self._apply_filters(
                    signal=signal,
                    confidence=confidence,
                    data=request.data,
                    model_config=model_config
                )
            
            # Lấy giá hiện tại
            current_price = None
            if not request.data.empty:
                current_price = request.data.iloc[-1]['close'] if 'close' in request.data.columns else None
            
            # Tạo kết quả
            result = InferenceResult(
                request_id=request.request_id,
                model_id=model_id,
                signal=signal,
                confidence=confidence,
                timestamp=datetime.now(),
                position_size=position_size,
                price=current_price,
                duration=model_config.parameters.get('target_duration', None),
                metadata={
                    'model_type': model_type.value if isinstance(model_type, ModelType) else model_type,
                    'features': list(features_data.shape) if hasattr(features_data, 'shape') else None,
                    'symbol': model_config.symbol,
                    'timeframe': model_config.timeframe
                }
            )
            
            # Cập nhật thống kê
            with self.model_lock:
                self.stats['successful_inferences'] += 1
                self.stats['models_usage'][model_id] += 1
                self.stats['signals_generated'][signal.value if isinstance(signal, SignalType) else signal.lower()] += 1
            
            self.logger.debug(f"Hoàn thành suy luận cho yêu cầu {request.request_id}, tín hiệu: {signal.value if isinstance(signal, SignalType) else signal}")
            
            return result
            
        except Exception as e:
            error_msg = f"Lỗi khi thực hiện suy luận cho mô hình {model_id}: {str(e)}"
            error_trace = traceback.format_exc()
            
            self.logger.error(f"{error_msg}\n{error_trace}")
            
            # Xử lý lỗi
            if self.error_handler:
                self.error_handler.handle_error(
                    error=e,
                    component="inference_engine.model",
                    category=ErrorCategory.MODEL,
                    severity=ErrorSeverity.MEDIUM,
                    context={"model_id": model_id, "request_id": request.request_id, "trace": error_trace}
                )
            
            # Cập nhật thống kê
            self.stats['failed_inferences'] += 1
            
            # Trả về kết quả lỗi
            return InferenceResult(
                request_id=request.request_id,
                model_id=model_id,
                signal=SignalType.HOLD,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={'error': error_msg, 'traceback': error_trace}
            )
    
    def _perform_ensemble_inference(self, request: InferenceRequest) -> InferenceResult:
        """
        Thực hiện suy luận ensemble với nhiều mô hình.
        
        Args:
            request: Yêu cầu suy luận
            
        Returns:
            Kết quả suy luận
        """
        model_id = request.model_id
        
        try:
            # Lấy mô hình và cấu hình
            with self.model_lock:
                model = self.models.get(model_id)
                model_config = self.model_configs.get(model_id)
            
            if not model or not model_config:
                error_msg = f"Không tìm thấy mô hình {model_id}"
                self.logger.error(error_msg)
                
                return InferenceResult(
                    request_id=request.request_id,
                    model_id=model_id,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': error_msg}
                )
            
            # Kiểm tra xem có phải mô hình ensemble không
            if not isinstance(model, dict) or model.get('type') != 'ensemble':
                error_msg = f"Mô hình {model_id} không phải là ensemble"
                self.logger.error(error_msg)
                
                return InferenceResult(
                    request_id=request.request_id,
                    model_id=model_id,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': error_msg}
                )
            
            # Lấy thông tin ensemble
            ensemble_models = model.get('models', {})
            ensemble_weights = model.get('weights', {})
            ensemble_method = model.get('method', 'average')
            
            if not ensemble_models:
                error_msg = f"Không có mô hình con trong ensemble {model_id}"
                self.logger.error(error_msg)
                
                return InferenceResult(
                    request_id=request.request_id,
                    model_id=model_id,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': error_msg}
                )
            
            # Tạo yêu cầu cho từng mô hình con
            sub_results = {}
            
            for sub_model_id, sub_model in ensemble_models.items():
                # Tạo yêu cầu con
                sub_request = InferenceRequest(
                    request_id=f"{request.request_id}_{sub_model_id}",
                    model_id=sub_model_id,
                    data=request.data,
                    timestamp=request.timestamp,
                    parameters=request.parameters,
                    priority=request.priority
                )
                
                # Tạo kết quả con
                sub_config = self.model_configs.get(sub_model_id)
                
                if sub_config:
                    # Thực hiện suy luận với mô hình con
                    sub_result = self._perform_inference(sub_request)
                    sub_results[sub_model_id] = sub_result
                else:
                    self.logger.warning(f"Không tìm thấy cấu hình cho mô hình con {sub_model_id}")
            
            # Tính toán kết quả tổng hợp
            if not sub_results:
                error_msg = f"Không có kết quả từ các mô hình con của ensemble {model_id}"
                self.logger.error(error_msg)
                
                return InferenceResult(
                    request_id=request.request_id,
                    model_id=model_id,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': error_msg}
                )
            
            # Tính tín hiệu và độ tin cậy
            if ensemble_method == 'average':
                # Tính trung bình có trọng số
                signal_counts = {s.value: 0 for s in SignalType}
                confidence_sum = 0.0
                weight_sum = 0.0
                
                for sub_model_id, sub_result in sub_results.items():
                    signal = sub_result.signal
                    if isinstance(signal, SignalType):
                        signal = signal.value
                    
                    confidence = sub_result.confidence
                    weight = ensemble_weights.get(sub_model_id, 1.0)
                    
                    signal_counts[signal] += weight
                    confidence_sum += confidence * weight
                    weight_sum += weight
                
                # Tìm tín hiệu có trọng số cao nhất
                final_signal = max(signal_counts.items(), key=lambda x: x[1])[0]
                final_signal = SignalType(final_signal)
                
                # Tính độ tin cậy trung bình có trọng số
                final_confidence = confidence_sum / weight_sum if weight_sum > 0 else 0.0
                
            elif ensemble_method == 'vote':
                # Bỏ phiếu đơn giản
                signal_votes = {}
                
                for sub_result in sub_results.values():
                    signal = sub_result.signal
                    if isinstance(signal, SignalType):
                        signal = signal.value
                    
                    if signal not in signal_votes:
                        signal_votes[signal] = 0
                    
                    signal_votes[signal] += 1
                
                # Tìm tín hiệu có số phiếu cao nhất
                final_signal = max(signal_votes.items(), key=lambda x: x[1])[0]
                final_signal = SignalType(final_signal)
                
                # Tính độ tin cậy dựa trên tỷ lệ phiếu
                final_confidence = signal_votes[final_signal.value] / len(sub_results)
                
            elif ensemble_method == 'highest_confidence':
                # Chọn kết quả có độ tin cậy cao nhất
                highest_conf_result = max(sub_results.values(), key=lambda x: x.confidence)
                
                final_signal = highest_conf_result.signal
                final_confidence = highest_conf_result.confidence
                
            else:
                # Mặc định là average
                self.logger.warning(f"Phương thức ensemble không được hỗ trợ: {ensemble_method}, sử dụng phương thức average")
                
                # Tính trung bình đơn giản
                signal_counts = {s.value: 0 for s in SignalType}
                total_confidence = 0.0
                
                for sub_result in sub_results.values():
                    signal = sub_result.signal
                    if isinstance(signal, SignalType):
                        signal = signal.value
                    
                    signal_counts[signal] += 1
                    total_confidence += sub_result.confidence
                
                # Tìm tín hiệu phổ biến nhất
                final_signal = max(signal_counts.items(), key=lambda x: x[1])[0]
                final_signal = SignalType(final_signal)
                
                # Tính độ tin cậy trung bình
                final_confidence = total_confidence / len(sub_results) if sub_results else 0.0
            
            # Kiểm tra ngưỡng tin cậy
            confidence_threshold = model_config.confidence_threshold
            
            if final_confidence < confidence_threshold:
                # Nếu độ tin cậy thấp, chuyển sang tín hiệu HOLD
                final_signal = SignalType.HOLD
                
                self.logger.debug(f"Độ tin cậy ensemble ({final_confidence:.4f}) dưới ngưỡng ({confidence_threshold}), chuyển sang HOLD")
            
            # Tính kích thước vị thế nếu cần
            position_size = None
            if final_signal in [SignalType.BUY, SignalType.SELL]:
                # Lấy cấu hình tín hiệu
                signal_config = self.config.get("signal_config", {})
                
                if signal_config.get("enable_autosizing", True):
                    position_size = self._calculate_position_size(
                        confidence=final_confidence,
                        signal=final_signal,
                        data=request.data,
                        risk_per_trade=signal_config.get("risk_per_trade", 0.02),
                        max_position_size=signal_config.get("max_position_size", 0.5)
                    )
                else:
                    position_size = signal_config.get("default_position_size", 0.1)
            
            # Áp dụng bộ lọc nếu cần
            if self.config.get("signal_config", {}).get("apply_filters", True):
                final_signal = self._apply_filters(
                    signal=final_signal,
                    confidence=final_confidence,
                    data=request.data,
                    model_config=model_config
                )
            
            # Lấy giá hiện tại
            current_price = None
            if not request.data.empty:
                current_price = request.data.iloc[-1]['close'] if 'close' in request.data.columns else None
            
            # Tạo kết quả cuối cùng
            result = InferenceResult(
                request_id=request.request_id,
                model_id=model_id,
                signal=final_signal,
                confidence=final_confidence,
                timestamp=datetime.now(),
                position_size=position_size,
                price=current_price,
                duration=model_config.parameters.get('target_duration', None),
                metadata={
                    'ensemble_method': ensemble_method,
                    'sub_models': len(sub_results),
                    'sub_results': {k: v.to_dict() for k, v in sub_results.items()},
                    'symbol': model_config.symbol,
                    'timeframe': model_config.timeframe
                }
            )
            
            # Cập nhật thống kê
            with self.model_lock:
                self.stats['successful_inferences'] += 1
                self.stats['models_usage'][model_id] += 1
                self.stats['signals_generated'][final_signal.value] += 1
            
            self.logger.debug(f"Hoàn thành suy luận ensemble cho yêu cầu {request.request_id}, tín hiệu: {final_signal.value}")
            
            return result
            
        except Exception as e:
            error_msg = f"Lỗi khi thực hiện suy luận ensemble cho mô hình {model_id}: {str(e)}"
            error_trace = traceback.format_exc()
            
            self.logger.error(f"{error_msg}\n{error_trace}")
            
            # Xử lý lỗi
            if self.error_handler:
                self.error_handler.handle_error(
                    error=e,
                    component="inference_engine.ensemble",
                    category=ErrorCategory.MODEL,
                    severity=ErrorSeverity.MEDIUM,
                    context={"model_id": model_id, "request_id": request.request_id, "trace": error_trace}
                )
            
            # Cập nhật thống kê
            self.stats['failed_inferences'] += 1
            
            # Trả về kết quả lỗi
            return InferenceResult(
                request_id=request.request_id,
                model_id=model_id,
                signal=SignalType.HOLD,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={'error': error_msg, 'traceback': error_trace}
            )
    
    def _perform_voting_inference(self, request: InferenceRequest) -> InferenceResult:
        """
        Thực hiện suy luận bằng phương pháp bỏ phiếu từ nhiều mô hình.
        
        Args:
            request: Yêu cầu suy luận
            
        Returns:
            Kết quả suy luận
        """
        # Lưu ý: Phương thức này có thể được triển khai tương tự như _perform_ensemble_inference
        # với một số điều chỉnh dành riêng cho việc bỏ phiếu
        # Ở đây đơn giản gọi _perform_ensemble_inference với method='vote'
        
        # Tạo copy của parameters và thêm method='vote'
        parameters = dict(request.parameters) if request.parameters else {}
        parameters['ensemble_method'] = 'vote'
        
        # Tạo yêu cầu mới
        voting_request = InferenceRequest(
            request_id=request.request_id,
            model_id=request.model_id,
            data=request.data,
            timestamp=request.timestamp,
            parameters=parameters,
            priority=request.priority,
            callback=request.callback
        )
        
        # Gọi _perform_ensemble_inference
        return self._perform_ensemble_inference(voting_request)
    
    def _perform_sequential_inference(self, request: InferenceRequest) -> InferenceResult:
        """
        Thực hiện suy luận tuần tự qua nhiều mô hình.
        
        Args:
            request: Yêu cầu suy luận
            
        Returns:
            Kết quả suy luận
        """
        model_id = request.model_id
        
        try:
            # Lấy mô hình và cấu hình
            with self.model_lock:
                model = self.models.get(model_id)
                model_config = self.model_configs.get(model_id)
            
            if not model or not model_config:
                error_msg = f"Không tìm thấy mô hình {model_id}"
                self.logger.error(error_msg)
                
                return InferenceResult(
                    request_id=request.request_id,
                    model_id=model_id,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': error_msg}
                )
            
            # Lấy danh sách mô hình con
            sequence = model_config.parameters.get('sequence', [])
            
            if not sequence:
                error_msg = f"Không có chuỗi mô hình cho sequential inference {model_id}"
                self.logger.error(error_msg)
                
                return InferenceResult(
                    request_id=request.request_id,
                    model_id=model_id,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': error_msg}
                )
            
            # Thực hiện suy luận tuần tự
            current_data = request.data
            sequence_results = []
            
            for i, step in enumerate(sequence):
                step_model_id = step.get('model_id')
                step_transform = step.get('transform')
                
                # Kiểm tra mô hình
                if step_model_id not in self.models:
                    self.logger.warning(f"Mô hình {step_model_id} trong chuỗi không tồn tại, bỏ qua")
                    continue
                
                # Tạo yêu cầu cho bước này
                step_request = InferenceRequest(
                    request_id=f"{request.request_id}_step{i}",
                    model_id=step_model_id,
                    data=current_data,
                    timestamp=request.timestamp,
                    parameters=step.get('parameters', {}),
                    priority=request.priority
                )
                
                # Thực hiện suy luận
                step_result = self._perform_inference(step_request)
                sequence_results.append(step_result)
                
                # Áp dụng biến đổi nếu cần
                if step_transform and callable(step_transform):
                    try:
                        current_data = step_transform(current_data, step_result)
                    except Exception as e:
                        self.logger.error(f"Lỗi khi áp dụng biến đổi: {str(e)}")
            
            # Lấy kết quả cuối cùng
            if not sequence_results:
                error_msg = "Không có kết quả từ chuỗi suy luận"
                self.logger.error(error_msg)
                
                return InferenceResult(
                    request_id=request.request_id,
                    model_id=model_id,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': error_msg}
                )
            
            # Lấy kết quả từ bước cuối cùng
            final_result = sequence_results[-1]
            
            # Cập nhật metadata
            final_result.metadata.update({
                'sequence_length': len(sequence),
                'sequence_results': [r.to_dict() for r in sequence_results[:-1]]  # Không bao gồm kết quả cuối
            })
            
            # Cập nhật thống kê
            with self.model_lock:
                self.stats['successful_inferences'] += 1
                self.stats['models_usage'][model_id] += 1
                self.stats['signals_generated'][final_result.signal.value if isinstance(final_result.signal, SignalType) else final_result.signal] += 1
            
            self.logger.debug(f"Hoàn thành suy luận tuần tự cho yêu cầu {request.request_id}")
            
            return final_result
            
        except Exception as e:
            error_msg = f"Lỗi khi thực hiện suy luận tuần tự cho mô hình {model_id}: {str(e)}"
            error_trace = traceback.format_exc()
            
            self.logger.error(f"{error_msg}\n{error_trace}")
            
            # Xử lý lỗi
            if self.error_handler:
                self.error_handler.handle_error(
                    error=e,
                    component="inference_engine.sequential",
                    category=ErrorCategory.MODEL,
                    severity=ErrorSeverity.MEDIUM,
                    context={"model_id": model_id, "request_id": request.request_id, "trace": error_trace}
                )
            
            # Cập nhật thống kê
            self.stats['failed_inferences'] += 1
            
            # Trả về kết quả lỗi
            return InferenceResult(
                request_id=request.request_id,
                model_id=model_id,
                signal=SignalType.HOLD,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={'error': error_msg, 'traceback': error_trace}
            )
    
    def _perform_hierarchical_inference(self, request: InferenceRequest) -> InferenceResult:
        """
        Thực hiện suy luận phân cấp với nhiều mô hình.
        
        Args:
            request: Yêu cầu suy luận
            
        Returns:
            Kết quả suy luận
        """
        model_id = request.model_id
        
        try:
            # Lấy mô hình và cấu hình
            with self.model_lock:
                model = self.models.get(model_id)
                model_config = self.model_configs.get(model_id)
            
            if not model or not model_config:
                error_msg = f"Không tìm thấy mô hình {model_id}"
                self.logger.error(error_msg)
                
                return InferenceResult(
                    request_id=request.request_id,
                    model_id=model_id,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': error_msg}
                )
            
            # Lấy cấu trúc phân cấp
            hierarchy = model_config.parameters.get('hierarchy', {})
            
            if not hierarchy:
                error_msg = f"Không có cấu trúc phân cấp cho hierarchical inference {model_id}"
                self.logger.error(error_msg)
                
                return InferenceResult(
                    request_id=request.request_id,
                    model_id=model_id,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': error_msg}
                )
            
            # Lấy mô hình gốc
            root_model_id = hierarchy.get('root')
            
            if not root_model_id:
                error_msg = "Không có mô hình gốc trong cấu trúc phân cấp"
                self.logger.error(error_msg)
                
                return InferenceResult(
                    request_id=request.request_id,
                    model_id=model_id,
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={'error': error_msg}
                )
            
            # Tạo yêu cầu cho mô hình gốc
            root_request = InferenceRequest(
                request_id=f"{request.request_id}_root",
                model_id=root_model_id,
                data=request.data,
                timestamp=request.timestamp,
                parameters=request.parameters,
                priority=request.priority
            )
            
            # Thực hiện suy luận với mô hình gốc
            root_result = self._perform_inference(root_request)
            
            # Kiểm tra xem có cần tiếp tục không
            root_signal = root_result.signal
            if isinstance(root_signal, SignalType):
                root_signal = root_signal.value
            
            # Lấy các mô hình con dựa trên kết quả của mô hình gốc
            children = hierarchy.get('children', {}).get(root_signal, [])
            
            if not children:
                # Nếu không có mô hình con, trả về kết quả của mô hình gốc
                self.logger.debug(f"Không có mô hình con cho tín hiệu {root_signal}, trả về kết quả của mô hình gốc")
                
                # Cập nhật thống kê
                with self.model_lock:
                    self.stats['successful_inferences'] += 1
                    self.stats['models_usage'][model_id] += 1
                    self.stats['signals_generated'][root_signal] += 1
                
                return root_result
            
            # Thực hiện suy luận với các mô hình con
            child_results = []
            
            for child_model_id in children:
                # Tạo yêu cầu cho mô hình con
                child_request = InferenceRequest(
                    request_id=f"{request.request_id}_{child_model_id}",
                    model_id=child_model_id,
                    data=request.data,
                    timestamp=request.timestamp,
                    parameters=request.parameters,
                    priority=request.priority
                )
                
                # Thực hiện suy luận
                child_result = self._perform_inference(child_request)
                child_results.append(child_result)
            
            # Tổng hợp kết quả
            if not child_results:
                # Nếu không có kết quả từ các mô hình con, trả về kết quả của mô hình gốc
                self.logger.debug("Không có kết quả từ các mô hình con, trả về kết quả của mô hình gốc")
                
                # Cập nhật thống kê
                with self.model_lock:
                    self.stats['successful_inferences'] += 1
                    self.stats['models_usage'][model_id] += 1
                    self.stats['signals_generated'][root_signal] += 1
                
                return root_result
            
            # Phương pháp tổng hợp
            merge_method = hierarchy.get('merge_method', 'highest_confidence')
            
            if merge_method == 'highest_confidence':
                # Chọn kết quả có độ tin cậy cao nhất
                best_result = max(child_results, key=lambda x: x.confidence)
                
                # Cập nhật metadata
                best_result.metadata.update({
                    'hierarchical': True,
                    'root_result': root_result.to_dict(),
                    'child_results': [r.to_dict() for r in child_results if r != best_result]
                })
                
                # Cập nhật thống kê
                with self.model_lock:
                    self.stats['successful_inferences'] += 1
                    self.stats['models_usage'][model_id] += 1
                    self.stats['signals_generated'][best_result.signal.value if isinstance(best_result.signal, SignalType) else best_result.signal] += 1
                
                self.logger.debug(f"Hoàn thành suy luận phân cấp cho yêu cầu {request.request_id}")
                
                return best_result
                
            elif merge_method == 'voting':
                # Bỏ phiếu
                signal_votes = {}
                total_confidence = 0.0
                
                for result in child_results:
                    signal = result.signal
                    if isinstance(signal, SignalType):
                        signal = signal.value
                    
                    if signal not in signal_votes:
                        signal_votes[signal] = 0
                    
                    signal_votes[signal] += 1
                    total_confidence += result.confidence
                
                # Tìm tín hiệu có số phiếu cao nhất
                best_signal = max(signal_votes.items(), key=lambda x: x[1])[0]
                best_signal = SignalType(best_signal)
                
                # Tính độ tin cậy
                confidence = total_confidence / len(child_results) if child_results else 0.0
                
                # Lấy giá hiện tại
                current_price = None
                if not request.data.empty:
                    current_price = request.data.iloc[-1]['close'] if 'close' in request.data.columns else None
                
                # Tạo kết quả
                result = InferenceResult(
                    request_id=request.request_id,
                    model_id=model_id,
                    signal=best_signal,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    price=current_price,
                    metadata={
                        'hierarchical': True,
                        'merge_method': 'voting',
                        'root_result': root_result.to_dict(),
                        'child_results': [r.to_dict() for r in child_results],
                        'votes': signal_votes
                    }
                )
                
                # Cập nhật thống kê
                with self.model_lock:
                    self.stats['successful_inferences'] += 1
                    self.stats['models_usage'][model_id] += 1
                    self.stats['signals_generated'][best_signal.value] += 1
                
                self.logger.debug(f"Hoàn thành suy luận phân cấp cho yêu cầu {request.request_id}")
                
                return result
                
            else:
                # Mặc định là highest_confidence
                self.logger.warning(f"Phương thức merge không được hỗ trợ: {merge_method}, sử dụng highest_confidence")
                
                # Chọn kết quả có độ tin cậy cao nhất
                best_result = max(child_results, key=lambda x: x.confidence)
                
                # Cập nhật metadata
                best_result.metadata.update({
                    'hierarchical': True,
                    'root_result': root_result.to_dict(),
                    'child_results': [r.to_dict() for r in child_results if r != best_result]
                })
                
                # Cập nhật thống kê
                with self.model_lock:
                    self.stats['successful_inferences'] += 1
                    self.stats['models_usage'][model_id] += 1
                    self.stats['signals_generated'][best_result.signal.value if isinstance(best_result.signal, SignalType) else best_result.signal] += 1
                
                self.logger.debug(f"Hoàn thành suy luận phân cấp cho yêu cầu {request.request_id}")
                
                return best_result
                
        except Exception as e:
            error_msg = f"Lỗi khi thực hiện suy luận phân cấp cho mô hình {model_id}: {str(e)}"
            error_trace = traceback.format_exc()
            
            self.logger.error(f"{error_msg}\n{error_trace}")
            
            # Xử lý lỗi
            if self.error_handler:
                self.error_handler.handle_error(
                    error=e,
                    component="inference_engine.hierarchical",
                    category=ErrorCategory.MODEL,
                    severity=ErrorSeverity.MEDIUM,
                    context={"model_id": model_id, "request_id": request.request_id, "trace": error_trace}
                )
            
            # Cập nhật thống kê
            self.stats['failed_inferences'] += 1
            
            # Trả về kết quả lỗi
            return InferenceResult(
                request_id=request.request_id,
                model_id=model_id,
                signal=SignalType.HOLD,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={'error': error_msg, 'traceback': error_trace}
            )
    
    def _prepare_features(self, data: pd.DataFrame, model_config: ModelConfig) -> np.ndarray:
        """
        Chuẩn bị các đặc trưng đầu vào cho mô hình.
        
        Args:
            data: DataFrame dữ liệu
            model_config: Cấu hình mô hình
            
        Returns:
            Mảng NumPy chứa đặc trưng đã chuẩn bị
        """
        input_features = model_config.input_features
        input_window = model_config.input_window
        
        try:
            # Kiểm tra xem có cần tạo đặc trưng bổ sung không
            if self.config.get("feature_config", {}).get("technical_indicators", {}).get("use_ta", False):
                # Tạo đặc trưng kỹ thuật
                feature_data = self.data_pipeline.generate_features(
                    data={model_config.symbol: data},
                    use_pipeline=None,
                    all_indicators=True
                )
                
                # Lấy DataFrame đã tạo đặc trưng
                if feature_data and model_config.symbol in feature_data:
                    data = feature_data[model_config.symbol]
            
            # Chọn các cột đặc trưng
            if input_features and all(feature in data.columns for feature in input_features):
                features_df = data[input_features]
            else:
                # Nếu không chỉ định hoặc không có sẵn, sử dụng tất cả
                features_df = data
            
            # Lấy cửa sổ dữ liệu
            if len(features_df) > input_window:
                window_data = features_df.iloc[-input_window:]
            else:
                # Nếu không đủ dữ liệu, sử dụng tất cả và padding
                window_data = features_df
                
                # Thêm padding nếu cần
                if len(window_data) < input_window:
                    padding_size = input_window - len(window_data)
                    padding = pd.DataFrame(0, index=range(padding_size), columns=window_data.columns)
                    window_data = pd.concat([padding, window_data], ignore_index=True)
            
            # Chuẩn hóa dữ liệu nếu cần
            normalization_config = self.config.get("feature_config", {}).get("normalization", {})
            normalization_method = normalization_config.get("method")
            
            if normalization_method == "min_max":
                # Chuẩn hóa Min-Max
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                window_data = pd.DataFrame(scaler.fit_transform(window_data), columns=window_data.columns)
                
            elif normalization_method == "z_score":
                # Chuẩn hóa Z-score
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                window_data = pd.DataFrame(scaler.fit_transform(window_data), columns=window_data.columns)
                
            elif normalization_method == "robust":
                # Chuẩn hóa robust
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                window_data = pd.DataFrame(scaler.fit_transform(window_data), columns=window_data.columns)
            
            # Chuyển đổi sang numpy array
            features = window_data.values
            
            # Điều chỉnh kích thước nếu cần
            model_type = model_config.model_type
            if isinstance(model_type, str):
                model_type = ModelType(model_type) if model_type in [m.value for m in ModelType] else ModelType.CUSTOM
            
            # Điều chỉnh dựa trên loại mô hình
            if model_type in [ModelType.DQN, ModelType.PPO, ModelType.A2C]:
                # Kiểm tra xem features có cần reshape không
                input_shape = model_config.parameters.get("input_shape")
                
                if input_shape:
                    # Reshape theo input_shape
                    try:
                        features = features.reshape(input_shape)
                    except Exception as e:
                        self.logger.error(f"Lỗi khi reshape features: {str(e)}")
                
                # Nếu không có input_shape, giữ nguyên kích thước
            
            return features
            
        except Exception as e:
            error_msg = f"Lỗi khi chuẩn bị đặc trưng: {str(e)}"
            error_trace = traceback.format_exc()
            
            self.logger.error(f"{error_msg}\n{error_trace}")
            
            # Trả về mảng trống trong trường hợp lỗi
            return np.array([])
    
    def _calculate_confidence(self, values: np.ndarray) -> float:
        """
        Tính độ tin cậy từ các giá trị đầu ra của mô hình.
        
        Args:
            values: Mảng giá trị đầu ra
            
        Returns:
            Độ tin cậy (0.0-1.0)
        """
        try:
            # Kiểm tra xem là Q-values hay probabilities
            if len(values.shape) > 0:
                # Đối với Q-values từ DQN
                if values.shape[0] > 1:
                    # Tính độ tin cậy dựa trên khoảng cách giữa các giá trị
                    max_value = values.max()
                    
                    if len(values) > 1:
                        second_max = np.partition(values.flatten(), -2)[-2]
                        confidence = (max_value - second_max) / (max_value - values.min() + 1e-6)
                        
                        # Giới hạn trong khoảng [0, 1]
                        confidence = max(0.0, min(1.0, confidence))
                    else:
                        confidence = 0.5  # Mặc định nếu chỉ có một giá trị
                else:
                    # Đối với một giá trị duy nhất
                    confidence = abs(values[0])
                    confidence = max(0.0, min(1.0, confidence))
            else:
                # Đối với một giá trị duy nhất
                confidence = abs(float(values))
                confidence = max(0.0, min(1.0, confidence))
                
            return confidence
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính độ tin cậy: {str(e)}")
            return 0.5  # Giá trị mặc định trong trường hợp lỗi
    
    def _convert_action_to_signal(self, action: Any, model_config: ModelConfig) -> SignalType:
        """
        Chuyển đổi hành động từ mô hình sang tín hiệu giao dịch.
        
        Args:
            action: Hành động từ mô hình
            model_config: Cấu hình mô hình
            
        Returns:
            Loại tín hiệu
        """
        try:
            # Đọc ánh xạ từ cấu hình
            action_map = model_config.parameters.get("action_map")
            
            if action_map and str(action) in action_map:
                # Sử dụng ánh xạ từ cấu hình
                signal_str = action_map[str(action)]
                
                try:
                    return SignalType(signal_str)
                except ValueError:
                    self.logger.warning(f"Loại tín hiệu không hợp lệ trong action_map: {signal_str}")
            
            # Sử dụng ánh xạ mặc định
            if action == 0:
                return SignalType.HOLD
            elif action == 1:
                return SignalType.BUY
            elif action == 2:
                return SignalType.SELL
            elif action == 3:
                return SignalType.CLOSE
            
            # Kiểm tra xem action có phải là chuỗi không
            if isinstance(action, str):
                try:
                    return SignalType(action.lower())
                except ValueError:
                    self.logger.warning(f"Loại tín hiệu không hợp lệ: {action}")
            
            # Mặc định
            self.logger.warning(f"Không thể chuyển đổi hành động {action} sang tín hiệu, sử dụng HOLD")
            return SignalType.HOLD
            
        except Exception as e:
            self.logger.error(f"Lỗi khi chuyển đổi hành động sang tín hiệu: {str(e)}")
            return SignalType.HOLD
    
    def _calculate_position_size(
        self,
        confidence: float,
        signal: SignalType,
        data: pd.DataFrame,
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.5
    ) -> float:
        """
        Tính kích thước vị thế dựa trên độ tin cậy và rủi ro.
        
        Args:
            confidence: Độ tin cậy của tín hiệu
            signal: Loại tín hiệu
            data: DataFrame dữ liệu
            risk_per_trade: Phần trăm rủi ro trên mỗi giao dịch
            max_position_size: Kích thước vị thế tối đa
            
        Returns:
            Kích thước vị thế (tỷ lệ của vốn)
        """
        try:
            # Cơ bản: Tỷ lệ dựa trên độ tin cậy
            base_size = risk_per_trade * confidence * 2  # Nhân 2 để scale lên
            
            # Giới hạn kích thước
            position_size = min(base_size, max_position_size)
            
            # Sử dụng volatility để điều chỉnh
            if not data.empty and 'close' in data.columns and len(data) > 20:
                # Tính volatility (độ lệch chuẩn của % thay đổi giá)
                returns = data['close'].pct_change().dropna()
                if len(returns) > 0:
                    volatility = returns.std()
                    
                    # Điều chỉnh kích thước dựa trên volatility
                    # Nếu volatility cao, giảm kích thước
                    volatility_factor = min(1.0, 0.01 / (volatility + 1e-6))
                    position_size *= volatility_factor
            
            # Đảm bảo lớn hơn 0
            position_size = max(0.01, position_size)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính kích thước vị thế: {str(e)}")
            return 0.1  # Giá trị mặc định trong trường hợp lỗi
    
    def _apply_filters(
        self,
        signal: SignalType,
        confidence: float,
        data: pd.DataFrame,
        model_config: ModelConfig
    ) -> SignalType:
        """
        Áp dụng các bộ lọc để cải thiện tín hiệu.
        
        Args:
            signal: Loại tín hiệu
            confidence: Độ tin cậy của tín hiệu
            data: DataFrame dữ liệu
            model_config: Cấu hình mô hình
            
        Returns:
            Loại tín hiệu sau khi lọc
        """
        # Nếu đã là HOLD, không cần lọc
        if signal == SignalType.HOLD:
            return signal
        
        try:
            # Lấy cấu hình bộ lọc
            filter_settings = self.config.get("signal_config", {}).get("filter_settings", {})
            
            # Lọc dựa trên ngưỡng tin cậy
            min_confidence = filter_settings.get("min_confidence", 0.6)
            if confidence < min_confidence:
                self.logger.debug(f"Tín hiệu bị lọc do độ tin cậy thấp: {confidence:.4f} < {min_confidence}")
                return SignalType.HOLD
            
            # Lọc dựa trên xu hướng
            if filter_settings.get("trend_filter", True) and not data.empty and 'close' in data.columns and len(data) > 50:
                # Tính SMA 50 và 200
                sma_50 = data['close'].rolling(window=50).mean()
                sma_200 = data['close'].rolling(window=200).mean()
                
                # Kiểm tra nếu đủ dữ liệu
                if not sma_50.empty and not sma_200.empty and not sma_50.iloc[-1] != sma_200.iloc[-1]:
                    # Xu hướng lên: SMA 50 > SMA 200
                    # Xu hướng xuống: SMA 50 < SMA 200
                    uptrend = sma_50.iloc[-1] > sma_200.iloc[-1]
                    
                    # Chỉ cho phép tín hiệu phù hợp với xu hướng
                    if signal == SignalType.BUY and not uptrend:
                        self.logger.debug("Tín hiệu BUY bị lọc do xu hướng xuống")
                        return SignalType.HOLD
                    elif signal == SignalType.SELL and uptrend:
                        self.logger.debug("Tín hiệu SELL bị lọc do xu hướng lên")
                        return SignalType.HOLD
            
            # Lọc dựa trên volatility
            if filter_settings.get("volatility_filter", True) and not data.empty and 'close' in data.columns and len(data) > 20:
                # Tính ATR (Average True Range)
                high = data['high'] if 'high' in data.columns else data['close']
                low = data['low'] if 'low' in data.columns else data['close']
                close = data['close']
                
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean()
                
                # Tính volatility như phần trăm của giá
                if not close.empty and close.iloc[-1] != 0:
                    volatility = atr.iloc[-1] / close.iloc[-1]
                    
                    # Nếu volatility quá cao, lọc tín hiệu
                    if volatility > 0.02:  # 2%
                        self.logger.debug(f"Tín hiệu bị lọc do volatility cao: {volatility:.4f}")
                        return SignalType.HOLD
            
            # Trả về tín hiệu gốc nếu vượt qua tất cả bộ lọc
            return signal
            
        except Exception as e:
            self.logger.error(f"Lỗi khi áp dụng bộ lọc: {str(e)}")
            return signal  # Trả về tín hiệu gốc trong trường hợp lỗi
    
    def _handle_inference_result(self, future: concurrent.futures.Future, request: InferenceRequest) -> None:
        """
        Xử lý kết quả suy luận từ worker.
        
        Args:
            future: Future từ worker
            request: Yêu cầu gốc
        """
        try:
            # Lấy kết quả
            result = future.result()
            
            # Lưu vào buffer kết quả
            with self.result_lock:
                self.results_buffer.append(result)
            
            # Gọi callback nếu có
            if request.callback and callable(request.callback):
                request.callback(result)
                
        except Exception as e:
            error_msg = f"Lỗi khi xử lý kết quả suy luận: {str(e)}"
            error_trace = traceback.format_exc()
            
            self.logger.error(f"{error_msg}\n{error_trace}")
            
            # Xử lý lỗi
            if self.error_handler:
                self.error_handler.handle_error(
                    error=e,
                    component="inference_engine.result_handler",
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.MEDIUM,
                    context={"request_id": request.request_id, "trace": error_trace}
                )
            
            # Cập nhật thống kê
            self.stats['failed_inferences'] += 1
            
            # Tạo kết quả lỗi
            error_result = InferenceResult(
                request_id=request.request_id,
                model_id=request.model_id,
                signal=SignalType.HOLD,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={'error': error_msg, 'traceback': error_trace}
            )
            
            # Lưu vào buffer kết quả
            with self.result_lock:
                self.results_buffer.append(error_result)
            
            # Gọi callback với kết quả lỗi
            if request.callback and callable(request.callback):
                request.callback(error_result)
    
    def submit_inference_request(
        self,
        data: pd.DataFrame,
        model_id: str,
        request_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 1,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Gửi yêu cầu suy luận mới.
        
        Args:
            data: DataFrame dữ liệu
            model_id: ID mô hình
            request_id: ID yêu cầu (sẽ tạo mới nếu không cung cấp)
            parameters: Các tham số bổ sung
            priority: Độ ưu tiên (1-10, càng cao càng ưu tiên)
            callback: Hàm callback khi hoàn thành
            
        Returns:
            ID yêu cầu
        """
        # Kiểm tra xem engine có đang chạy không
        if not self.is_running:
            raise RuntimeError("InferenceEngine không đang chạy")
        
        # Tạo ID yêu cầu nếu không cung cấp
        if request_id is None:
            request_id = f"req_{int(time.time())}_{model_id}_{hash(str(datetime.now().microsecond))}"
        
        # Tạo yêu cầu
        request = InferenceRequest(
            request_id=request_id,
            model_id=model_id,
            data=data,
            timestamp=datetime.now(),
            parameters=parameters,
            priority=priority,
            callback=callback
        )
        
        # Thêm vào queue
        self.request_queue.put(request)
        
        self.logger.debug(f"Đã gửi yêu cầu suy luận {request_id} cho mô hình {model_id}")
        
        return request_id
    
    def update_data_buffer(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Cập nhật buffer dữ liệu cho một symbol.
        
        Args:
            symbol: Symbol cập nhật
            data: DataFrame dữ liệu mới
            
        Returns:
            True nếu thành công, False nếu không
        """
        try:
            with self.buffer_lock:
                # Tạo buffer nếu chưa có
                if symbol not in self.data_buffers:
                    self.data_buffers[symbol] = deque(maxlen=self.data_buffer_size)
                
                # Thêm data vào buffer
                self.data_buffers[symbol].append(data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi cập nhật buffer dữ liệu cho {symbol}: {str(e)}")
            return False
    
    def get_data_buffer(self, symbol: str, length: Optional[int] = None) -> pd.DataFrame:
        """
        Lấy dữ liệu từ buffer.
        
        Args:
            symbol: Symbol cần lấy dữ liệu
            length: Số lượng dữ liệu muốn lấy (None để lấy tất cả)
            
        Returns:
            DataFrame dữ liệu
        """
        try:
            with self.buffer_lock:
                if symbol not in self.data_buffers:
                    self.logger.warning(f"Không có dữ liệu cho {symbol} trong buffer")
                    return pd.DataFrame()
                
                buffer = self.data_buffers[symbol]
                
                if not buffer:
                    return pd.DataFrame()
                
                # Giới hạn số lượng nếu cần
                if length is not None and length > 0:
                    data_list = list(buffer)[-length:]
                else:
                    data_list = list(buffer)
                
                # Ghép các DataFrame
                return pd.concat(data_list, ignore_index=True)
                
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy dữ liệu từ buffer cho {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_results(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Lấy danh sách kết quả suy luận gần đây.
        
        Args:
            count: Số lượng kết quả muốn lấy
            
        Returns:
            Danh sách kết quả
        """
        try:
            with self.result_lock:
                latest_results = list(self.results_buffer)[-count:]
                return [result.to_dict() for result in latest_results]
                
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy kết quả gần đây: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê hiệu suất.
        
        Returns:
            Dict chứa thông kê
        """
        return {
            'stats': dict(self.stats),
            'performance': dict(self.performance_metrics),
            'status': {
                'is_running': self.is_running,
                'worker_threads': self.worker_threads,
                'queue_size': self.request_queue.qsize(),
                'models_count': len(self.models),
                'data_buffers_count': len(self.data_buffers),
                'results_buffer_size': len(self.results_buffer)
            }
        }
    
    def healthcheck(self) -> Dict[str, Any]:
        """
        Kiểm tra sức khỏe của engine.
        
        Returns:
            Dict chứa kết quả kiểm tra
        """
        # Kiểm tra xem engine có đang chạy không
        is_running = self.is_running
        
        # Kiểm tra thread xử lý
        processor_alive = self.processor_thread.is_alive() if self.processor_thread else False
        
        # Kiểm tra thread giám sát
        monitor_alive = self.monitor_thread.is_alive() if self.monitor_thread else False
        
        # Kiểm tra queue
        queue_size = self.request_queue.qsize()
        
        # Kiểm tra mô hình
        models_count = len(self.models)
        
        # Kiểm tra worker pool
        worker_pool_running = self.worker_pool is not None and not self.worker_pool._shutdown
        
        # Kiểm tra xem có bị quá tải không
        is_overloaded = queue_size > 100 or self.performance_metrics['error_rate'] > 0.3
        
        # Tính trạng thái sức khỏe
        if is_running and processor_alive and monitor_alive and worker_pool_running and not is_overloaded:
            health_status = "healthy"
        elif is_running and (not processor_alive or not monitor_alive or not worker_pool_running):
            health_status = "degraded"
        elif is_overloaded:
            health_status = "overloaded"
        elif not is_running:
            health_status = "stopped"
        else:
            health_status = "unknown"
        
        return {
            'status': health_status,
            'timestamp': datetime.now().isoformat(),
            'is_running': is_running,
            'processor_alive': processor_alive,
            'monitor_alive': monitor_alive,
            'worker_pool_running': worker_pool_running,
            'queue_size': queue_size,
            'models_count': models_count,
            'error_rate': self.performance_metrics['error_rate'],
            'requests_per_minute': self.performance_metrics['requests_per_minute'],
            'avg_processing_time': self.stats['avg_processing_time']
        }

# Tạo API HTTP cho InferenceEngine (có thể làm)
def create_inference_api(inference_engine, host="localhost", port=8002):
    """
    Tạo API HTTP đơn giản cho InferenceEngine.
    """
    # Chỗ này có thể sử dụng Flask hoặc FastAPI để tạo API
    pass