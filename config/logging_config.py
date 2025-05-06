"""
Cấu hình logging hệ thống.
File này thiết lập cấu hình logging cho toàn bộ ứng dụng, bao gồm
các định dạng log, mức độ log, và handlers cho các thành phần khác nhau.
"""

import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from config.system_config import LOG_DIR, get_system_config

# Lấy cấu hình hệ thống
system_config = get_system_config()

# Đường dẫn file log
LOG_FILE_PATH = LOG_DIR / f"trading_system_{datetime.now().strftime('%Y%m%d')}.log"
ERROR_LOG_PATH = LOG_DIR / f"error_{datetime.now().strftime('%Y%m%d')}.log"
EXCHANGE_LOG_DIR = LOG_DIR / "exchanges"

# Thư mục log cho huấn luyện
TRAINING_LOG_DIR = LOG_DIR / "training"

# Đảm bảo các thư mục logs tồn tại
LOG_DIR.mkdir(exist_ok=True, parents=True)
EXCHANGE_LOG_DIR.mkdir(exist_ok=True, parents=True)
TRAINING_LOG_DIR.mkdir(exist_ok=True, parents=True)

# Định dạng log cho huấn luyện
TRAINING_LOG_FORMAT = "📈 [EP %(episode)s] Reward: %(reward).2f | Winrate: %(winrate).2f | Loss: %(loss).4f | KL: %(kl).4f | Entropy: %(entropy).4f"
EPISODE_SUMMARY_FORMAT = "🔄 [Episode %(episode)s/%(total_episodes)s] Reward: %(reward).2f | Steps: %(steps)d | Win: %(wins)d | Loss: %(losses)d | Time: %(time).1fs"
EVAL_FORMAT = "🔍 [Evaluation] Mean: %(mean).2f | Min: %(min).2f | Max: %(max).2f | Std: %(std).2f | Win: %(win).2f%%"

# Tên logger cho các module khác nhau
LOGGER_NAMES = {
    "main": "trading_system",
    "data_collector": "trading_system.data_collector",
    "data_processor": "trading_system.data_processor",
    "model": "trading_system.model",
    "environment": "trading_system.environment",
    "risk_management": "trading_system.risk_management",
    "backtest": "trading_system.backtest",
    "deployment": "trading_system.deployment",
    "api": "trading_system.api",
    "security": "trading_system.security",
    "exchange": "trading_system.exchange",
}

# Thêm logger cho huấn luyện
LOGGER_NAMES.update({
    "training": "trading_system.training",
    "training_metrics": "trading_system.training.metrics",
    "training_summary": "trading_system.training.summary",
    "agent": "trading_system.agent",
    "evaluation": "trading_system.evaluation",
})

# Định dạng log mặc định
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAIL_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

# Mapping mức độ log
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Lấy mức độ log từ cấu hình hoặc mặc định
log_level_name = system_config.get("log_level", "INFO")
LOG_LEVEL = LOG_LEVEL_MAP.get(log_level_name, logging.INFO)

class TrainingLogFilter(logging.Filter):
    """
    Filter cho các log huấn luyện để thêm thông tin bổ sung.
    """
    def filter(self, record):
        # Đảm bảo các thuộc tính tồn tại
        if not hasattr(record, 'episode'):
            record.episode = 0
        if not hasattr(record, 'reward'):
            record.reward = 0.0
        if not hasattr(record, 'winrate'):
            record.winrate = 0.0
        if not hasattr(record, 'loss'):
            record.loss = 0.0
        if not hasattr(record, 'kl'):
            record.kl = 0.0
        if not hasattr(record, 'entropy'):
            record.entropy = 0.0
        return True

class TrainingLogFormatter(logging.Formatter):
    """
    Formatter đặc biệt cho log huấn luyện.
    """
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
    
    def format(self, record):
        # Nếu có thuộc tính log_type và giá trị là 'training'
        if hasattr(record, 'log_type') and record.log_type == 'training':
            return TRAINING_LOG_FORMAT % record.__dict__
        else:
            return super().format(record)

class LoggingConfig:
    """
    Lớp quản lý cấu hình logging.
    Cung cấp phương thức thiết lập và cấu hình các logger khác nhau.
    """
    
    def __init__(self):
        self.loggers = {}
        self.handlers = {}
        self.formatters = {}
        
        # Tạo các formatter
        self.create_formatters()
        
        # Tạo các handler mặc định
        self.create_handlers()
    
    def create_formatters(self) -> None:
        """Tạo các formatter cho log."""
        self.formatters["default"] = logging.Formatter(DEFAULT_LOG_FORMAT)
        self.formatters["detailed"] = logging.Formatter(DETAIL_LOG_FORMAT)
        self.formatters["training"] = TrainingLogFormatter(TRAINING_LOG_FORMAT)
    
    def create_handlers(self) -> None:
        """Tạo các handler cho log."""
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatters["default"])
        console_handler.setLevel(LOG_LEVEL)
        self.handlers["console"] = console_handler
        
        # File handler cho tất cả các log
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE_PATH, 
            maxBytes=10_000_000,  # ~10MB
            backupCount=10,
            encoding="utf-8"
        )
        file_handler.setFormatter(self.formatters["detailed"])
        file_handler.setLevel(LOG_LEVEL)
        self.handlers["file"] = file_handler
        
        # File handler cho các error
        error_handler = logging.handlers.RotatingFileHandler(
            ERROR_LOG_PATH, 
            maxBytes=10_000_000,  # ~10MB
            backupCount=10,
            encoding="utf-8"
        )
        error_handler.setFormatter(self.formatters["detailed"])
        error_handler.setLevel(logging.ERROR)
        self.handlers["error_file"] = error_handler
        
        # Thêm handler cho training logs
        training_console_handler = logging.StreamHandler()
        training_console_handler.setFormatter(self.formatters["training"])
        training_console_handler.setLevel(LOG_LEVEL)
        training_console_handler.addFilter(TrainingLogFilter())
        self.handlers["training_console"] = training_console_handler
        
        # File handler cho training logs
        training_file_handler = logging.handlers.RotatingFileHandler(
            TRAINING_LOG_DIR / f"training_{datetime.now().strftime('%Y%m%d')}.log", 
            maxBytes=10_000_000,  # ~10MB
            backupCount=5,
            encoding="utf-8"
        )
        training_file_handler.setFormatter(self.formatters["training"])
        training_file_handler.setLevel(LOG_LEVEL)
        training_file_handler.addFilter(TrainingLogFilter())
        self.handlers["training_file"] = training_file_handler

        
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Lấy logger theo tên.
        
        Args:
            name: Tên của logger (được định nghĩa trong LOGGER_NAMES hoặc tên tùy chỉnh)
            
        Returns:
            Logger được cấu hình
        """
        if name in self.loggers:
            return self.loggers[name]
        
        if name in LOGGER_NAMES:
            logger_name = LOGGER_NAMES[name]
        else:
            # Kiểm tra xem có phải exchange connector không
            if name.endswith("_connector"):
                logger_name = f"trading_system.exchange.{name}"
            else:
                logger_name = f"trading_system.{name}"
        
        logger = logging.getLogger(logger_name)
        logger.setLevel(LOG_LEVEL)
        
        # Xóa các handler cũ (nếu có)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Thêm các handler mặc định
        logger.addHandler(self.handlers["console"])
        logger.addHandler(self.handlers["file"])
        logger.addHandler(self.handlers["error_file"])
        
        # Nếu là exchange connector, thêm file handler riêng
        if name.endswith("_connector"):
            exchange_file_handler = logging.handlers.RotatingFileHandler(
                EXCHANGE_LOG_DIR / f"{name}.log",
                maxBytes=5_000_000,  # ~5MB
                backupCount=5,
                encoding="utf-8"
            )
            exchange_file_handler.setFormatter(self.formatters["detailed"])
            exchange_file_handler.setLevel(LOG_LEVEL)
            logger.addHandler(exchange_file_handler)
        
        # Đảm bảo không lan truyền log lên logger cha
        logger.propagate = False
        
        self.loggers[name] = logger
        return logger
    
    def get_training_logger(self, name="training") -> logging.Logger:
        """
        Lấy logger đặc biệt cho huấn luyện.
        
        Args:
            name: Tên của logger (thường là "training")
            
        Returns:
            Logger được cấu hình đặc biệt cho huấn luyện
        """
        logger_name = LOGGER_NAMES.get(name, f"trading_system.{name}")
        logger = logging.getLogger(logger_name)
        logger.setLevel(LOG_LEVEL)
        
        # Xóa các handler cũ (nếu có)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Thêm các handler của training
        logger.addHandler(self.handlers["training_console"])
        logger.addHandler(self.handlers["training_file"])
        logger.addHandler(self.handlers["error_file"])
        
        # Đảm bảo không lan truyền log lên logger cha
        logger.propagate = False
        
        self.loggers[name] = logger
        return logger
    
    def set_level(self, name: str, level: str) -> None:
        """
        Đặt mức độ log cho một logger cụ thể.
        
        Args:
            name: Tên của logger
            level: Tên mức độ log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if name in self.loggers:
            logger = self.loggers[name]
            log_level = LOG_LEVEL_MAP.get(level, logging.INFO)
            logger.setLevel(log_level)
    
    def set_global_level(self, level: str) -> None:
        """
        Đặt mức độ log cho tất cả các logger.
        
        Args:
            level: Tên mức độ log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        log_level = LOG_LEVEL_MAP.get(level, logging.INFO)
        
        # Cập nhật mức độ log cho các handler
        for handler in self.handlers.values():
            if handler != self.handlers["error_file"]:  # Giữ mức độ ERROR cho error_file
                handler.setLevel(log_level)
        
        # Cập nhật mức độ log cho các logger
        for logger in self.loggers.values():
            logger.setLevel(log_level)
    
    def add_custom_handler(self, name: str, handler: logging.Handler, 
                           logger_names: Optional[list] = None) -> None:
        """
        Thêm handler tùy chỉnh cho các logger.
        
        Args:
            name: Tên của handler
            handler: Handler cần thêm
            logger_names: Danh sách tên logger cần thêm handler (None để thêm vào tất cả)
        """
        self.handlers[name] = handler
        
        # Thêm handler vào các logger cụ thể hoặc tất cả
        if logger_names is None:
            logger_names = list(self.loggers.keys())
            
        for logger_name in logger_names:
            if logger_name in self.loggers:
                self.loggers[logger_name].addHandler(handler)

# Tạo instance mặc định để sử dụng trong ứng dụng
logging_config = LoggingConfig()

def get_logger(name: str) -> logging.Logger:
    """
    Hàm helper chính để lấy logger đã được cấu hình.
    Được sử dụng ở hầu hết các module của hệ thống.
    
    Args:
        name: Tên của logger (định nghĩa trong LOGGER_NAMES hoặc tên tùy chỉnh)
        
    Returns:
        Logger được cấu hình
    """
    return logging_config.get_logger(name)

def get_training_logger(name="training") -> logging.Logger:
    """
    Lấy logger được cấu hình đặc biệt cho huấn luyện agent.
    
    Args:
        name: Tên module ("training", "agent", "evaluation")
        
    Returns:
        Logger đặc biệt cho huấn luyện
    """
    return logging_config.get_training_logger(name)

def log_training_metrics(logger, episode, reward, winrate, loss=0.0, kl=0.0, entropy=0.0):
    """
    Helper để log metrics huấn luyện với format đẹp.
    
    Args:
        logger: Logger đối tượng
        episode: Số episode hiện tại
        reward: Giá trị phần thưởng
        winrate: Tỷ lệ thắng 
        loss: Giá trị loss
        kl: Giá trị KL divergence
        entropy: Giá trị entropy
    """
    record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None
    )
    
    # Thêm các thuộc tính đặc biệt
    record.log_type = "training"
    record.episode = episode
    record.reward = reward
    record.winrate = winrate
    record.loss = loss
    record.kl = kl
    record.entropy = entropy
    
    logger.handle(record)

def log_episode_summary(logger, episode, total_episodes, steps, reward, wins, losses, elapsed_time):
    """
    Helper để log tóm tắt episode.
    
    Args:
        logger: Logger đối tượng
        episode: Số episode hiện tại
        total_episodes: Tổng số episode cần huấn luyện
        steps: Số bước trong episode
        reward: Phần thưởng tổng
        wins: Số lần thắng
        losses: Số lần thua
        elapsed_time: Thời gian thực hiện (giây)
    """
    record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None
    )
    
    # Thêm các thuộc tính đặc biệt
    record.log_type = "episode_summary"
    record.episode = episode
    record.total_episodes = total_episodes
    record.steps = steps
    record.reward = reward
    record.wins = wins
    record.losses = losses
    record.time = elapsed_time
    
    logger.handle(record)

def log_evaluation_results(logger, mean, min_val, max_val, std, win_rate):
    """
    Helper để log kết quả đánh giá.
    
    Args:
        logger: Logger đối tượng
        mean: Phần thưởng trung bình
        min_val: Phần thưởng nhỏ nhất
        max_val: Phần thưởng lớn nhất
        std: Độ lệch chuẩn
        win_rate: Tỷ lệ thắng (0.0-1.0)
    """
    record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None
    )
    
    # Thêm các thuộc tính đặc biệt
    record.log_type = "evaluation"
    record.mean = mean
    record.min = min_val
    record.max = max_val
    record.std = std
    record.win = win_rate * 100  # Chuyển thành phần trăm
    
    logger.handle(record)

def setup_logger(name: str) -> logging.Logger:
    """
    Hàm helper dành riêng cho các exchange connectors.
    Tạo và cấu hình logger với file log riêng cho mỗi exchange.
    
    QUAN TRỌNG: Hàm này được sử dụng trong các file:
    - data_collectors/exchange_api/generic_connector.py
    - data_collectors/exchange_api/binance_connector.py
    - data_collectors/exchange_api/bybit_connector.py
    
    Args:
        name: Tên của logger (thường là "{exchange_id}_connector")
        
    Returns:
        Logger được cấu hình
    """
    return get_logger(name)

def setup_component_logger(component_name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Helper để cài đặt logger cho một thành phần cụ thể với mức log tùy chỉnh.
    Thường được sử dụng cho các module lớn của hệ thống.
    
    Args:
        component_name: Tên thành phần
        level: Mức độ log (None để sử dụng mức mặc định)
    
    Returns:
        Logger được cấu hình cho thành phần
    """
    logger = get_logger(component_name)
    
    if level and level in LOG_LEVEL_MAP:
        logger.setLevel(LOG_LEVEL_MAP[level])
    
    return logger