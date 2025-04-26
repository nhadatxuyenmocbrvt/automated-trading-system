"""
Cấu hình chung hệ thống.
File này chứa các cài đặt và cấu hình hệ thống cơ bản như đường dẫn, thông số,
và các cài đặt mặc định cho toàn bộ dự án.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Đường dẫn thư mục gốc của dự án
BASE_DIR = Path(__file__).parent.parent

# Các đường dẫn thư mục con
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"
LOG_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"
TEMP_DIR = BASE_DIR / "temp"

# Tự động tạo các thư mục nếu chưa tồn tại
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Thông tin phiên bản
VERSION = "0.1.0"
BUILD_DATE = datetime.now().strftime("%Y-%m-%d")

# Cấu hình chung cho hệ thống
SYSTEM_CONFIG = {
    "version": VERSION,
    "build_date": BUILD_DATE,
    "environment": os.getenv("TRADING_ENV", "development"),
    "debug_mode": os.getenv("DEBUG_MODE", "True").lower() in ("true", "1", "t"),
    "max_threads": int(os.getenv("MAX_THREADS", "4")),
    "max_processes": int(os.getenv("MAX_PROCESSES", "2")),
    "request_timeout": int(os.getenv("REQUEST_TIMEOUT", "30")),  # seconds
    "max_retries": int(os.getenv("MAX_RETRIES", "3")),
    "memory_limit": int(os.getenv("MEMORY_LIMIT", "8192")),  # MB
    "data_storage_format": os.getenv("DATA_STORAGE_FORMAT", "csv"),
    "default_exchange": os.getenv("DEFAULT_EXCHANGE", "binance"),
}

# Cấu hình môi trường
ENVIRONMENT_CONFIGS = {
    "development": {
        "debug_mode": True,
        "log_level": "DEBUG",
        "use_mocks": True,
        "db_uri": "sqlite:///dev.db",
    },
    "testing": {
        "debug_mode": True,
        "log_level": "INFO",
        "use_mocks": True,
        "db_uri": "sqlite:///test.db",
    },
    "production": {
        "debug_mode": False,
        "log_level": "WARNING",
        "use_mocks": False,
        "db_uri": os.getenv("DATABASE_URI", "sqlite:///prod.db"),
    }
}

# Cấu hình cho các công cụ giao dịch
TRADING_CONFIG = {
    "default_symbol": "BTC/USDT",
    "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
    "default_timeframe": "1h",
    "lookback_periods": int(os.getenv("LOOKBACK_PERIODS", "100")),
    "order_types": ["market", "limit", "stop_loss", "take_profit"],
    "default_order_type": "limit",
    "max_open_positions": int(os.getenv("MAX_OPEN_POSITIONS", "5")),
}

# Cấu hình cho các agent
AGENT_CONFIG = {
    "default_agent": "dqn_agent",
    "random_seed": int(os.getenv("RANDOM_SEED", "42")),
    "save_frequency": int(os.getenv("SAVE_FREQUENCY", "100")),  # episodes
    "eval_frequency": int(os.getenv("EVAL_FREQUENCY", "50")),  # episodes
    "checkpoint_dir": str(MODEL_DIR / "checkpoints"),
    "tensorboard_log_dir": str(LOG_DIR / "tensorboard"),
}

class SystemConfig:
    """
    Lớp quản lý cấu hình hệ thống.
    Cung cấp phương thức truy cập và cập nhật cấu hình.
    """
    def __init__(self):
        self.config = SYSTEM_CONFIG.copy()
        # Tải cấu hình môi trường hiện tại
        env_name = self.config["environment"]
        if env_name in ENVIRONMENT_CONFIGS:
            self.config.update(ENVIRONMENT_CONFIGS[env_name])
        
        # Tải các cấu hình khác
        self.config["trading"] = TRADING_CONFIG.copy()
        self.config["agent"] = AGENT_CONFIG.copy()
        
        # Tạo thư mục checkpoints nếu chưa tồn tại
        Path(self.config["agent"]["checkpoint_dir"]).mkdir(exist_ok=True, parents=True)
        Path(self.config["agent"]["tensorboard_log_dir"]).mkdir(exist_ok=True, parents=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Lấy giá trị cấu hình theo khóa."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Cập nhật giá trị cấu hình."""
        keys = key.split('.')
        config = self.config
        
        # Duyệt đến cấp trước cấp cuối
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def load_from_file(self, file_path: str) -> None:
        """Tải cấu hình từ file JSON."""
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
            self.config.update(loaded_config)
    
    def save_to_file(self, file_path: str) -> None:
        """Lưu cấu hình ra file JSON."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
    
    def get_all(self) -> Dict[str, Any]:
        """Lấy toàn bộ cấu hình."""
        return self.config.copy()
    
    def reset(self) -> None:
        """Đặt lại cấu hình về mặc định."""
        self.__init__()

# Tạo instance mặc định để sử dụng trong ứng dụng
system_config = SystemConfig()

def get_system_config() -> SystemConfig:
    """Hàm helper để lấy instance SystemConfig."""
    return system_config