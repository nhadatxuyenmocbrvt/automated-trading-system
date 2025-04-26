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

# Đảm bảo thư mục logs tồn tại
LOG_DIR.mkdir(exist_ok=True, parents=True)

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
}

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
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Lấy logger theo tên.
        
        Args:
            name: Tên của logger (được định nghĩa trong LOGGER_NAMES)
            
        Returns:
            Logger được cấu hình
        """
        if name in self.loggers:
            return self.loggers[name]
        
        if name in LOGGER_NAMES:
            logger_name = LOGGER_NAMES[name]
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
    Hàm helper để lấy logger đã được cấu hình.
    
    Args:
        name: Tên của logger (định nghĩa trong LOGGER_NAMES hoặc tên tùy chỉnh)
        
    Returns:
        Logger được cấu hình
    """
    return logging_config.get_logger(name)

def setup_component_logger(component_name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Helper để cài đặt logger cho một thành phần cụ thể.
    
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