"""
Quản lý biến môi trường.
File này cung cấp các hàm để tải và truy cập biến môi trường từ file .env,
cùng với việc xác thực các biến bắt buộc.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from dotenv import load_dotenv

from config.system_config import BASE_DIR

# Các loại biến môi trường
ENV_TYPES = {
    "str": str,
    "int": int,
    "float": float,
    "bool": lambda v: v.lower() in ("true", "1", "t", "yes", "y"),
    "json": json.loads,
    "list": lambda v: v.split(",") if v else []
}

# Định nghĩa biến môi trường
ENV_VARS = {
    # Biến hệ thống
    "TRADING_ENV": {"type": "str", "default": "development", "description": "Môi trường hệ thống (development, testing, production)"},
    "DEBUG_MODE": {"type": "bool", "default": "True", "description": "Chế độ debug"},
    "LOG_LEVEL": {"type": "str", "default": "INFO", "description": "Mức độ log (DEBUG, INFO, WARNING, ERROR, CRITICAL)"},
    "DATABASE_URI": {"type": "str", "default": "", "description": "URI kết nối database"},
    
    # API và bảo mật
    "API_PORT": {"type": "int", "default": "5000", "description": "Cổng API hệ thống"},
    "API_HOST": {"type": "str", "default": "127.0.0.1", "description": "Host API hệ thống"},
    "SECRET_KEY": {"type": "str", "default": "", "description": "Khóa bí mật cho session/token", "required": True},
    "TOKEN_EXPIRY": {"type": "int", "default": "86400", "description": "Thời gian hết hạn token (giây)"},
    "SECURITY_LEVEL": {"type": "str", "default": "high", "description": "Mức độ bảo mật (low, medium, high)"},
    
    # Sàn giao dịch
    "DEFAULT_EXCHANGE": {"type": "str", "default": "binance", "description": "Sàn giao dịch mặc định"},
    "BINANCE_API_KEY": {"type": "str", "default": "", "description": "API Key Binance"},
    "BINANCE_API_SECRET": {"type": "str", "default": "", "description": "API Secret Binance"},
    "BYBIT_API_KEY": {"type": "str", "default": "", "description": "API Key ByBit"},
    "BYBIT_API_SECRET": {"type": "str", "default": "", "description": "API Secret ByBit"},
    
    # Cấu hình giao dịch
    "MAX_OPEN_POSITIONS": {"type": "int", "default": "5", "description": "Số vị thế mở tối đa"},
    "DEFAULT_LEVERAGE": {"type": "float", "default": "1.0", "description": "Đòn bẩy mặc định"},
    "RISK_PER_TRADE": {"type": "float", "default": "0.02", "description": "Rủi ro mỗi giao dịch (%)"},
    "ASSET_LIST": {"type": "list", "default": "BTC/USDT,ETH/USDT", "description": "Danh sách tài sản giao dịch"},
    
    # Huấn luyện agent
    "RANDOM_SEED": {"type": "int", "default": "42", "description": "Seed cho số ngẫu nhiên"},
    "BATCH_SIZE": {"type": "int", "default": "64", "description": "Kích thước batch huấn luyện"},
    "LEARNING_RATE": {"type": "float", "default": "0.001", "description": "Tốc độ học"},
    "GAMMA": {"type": "float", "default": "0.99", "description": "Hệ số giảm phần thưởng"},
    "EPSILON": {"type": "float", "default": "0.1", "description": "Hệ số epsilon cho exploration"},
    "TRAIN_EPISODES": {"type": "int", "default": "1000", "description": "Số episode huấn luyện"},
    
    # Performance
    "MAX_THREADS": {"type": "int", "default": "4", "description": "Số luồng tối đa"},
    "MAX_PROCESSES": {"type": "int", "default": "2", "description": "Số tiến trình tối đa"},
    "REQUEST_TIMEOUT": {"type": "int", "default": "30", "description": "Thời gian timeout request (giây)"},
    "MAX_RETRIES": {"type": "int", "default": "3", "description": "Số lần thử lại tối đa"},
    "MEMORY_LIMIT": {"type": "int", "default": "8192", "description": "Giới hạn bộ nhớ (MB)"},
}

class EnvManager:
    """
    Lớp quản lý biến môi trường.
    """
    
    def __init__(self, env_file: Optional[str] = None):
        self.env_file = env_file
        self.env_values = {}
        self.load_env()
    
    def load_env(self) -> None:
        """
        Tải biến môi trường từ file .env nếu tồn tại,
        hoặc sử dụng các biến môi trường hệ thống.
        """
        # Đường dẫn mặc định tới file .env
        env_path = BASE_DIR / ".env"
        
        # Sử dụng file tùy chỉnh nếu được chỉ định
        if self.env_file:
            env_path = Path(self.env_file)
        
        # Tải file .env nếu tồn tại
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"Đã tải biến môi trường từ {env_path}")
        
        # Đặt các giá trị mặc định nếu biến môi trường không tồn tại
        for var_name, config in ENV_VARS.items():
            self.env_values[var_name] = self._get_env_var(var_name, config)
    
    def _get_env_var(self, var_name: str, config: Dict[str, Any]) -> Any:
        """
        Lấy giá trị biến môi trường với kiểu dữ liệu đúng.
        
        Args:
            var_name: Tên biến môi trường
            config: Cấu hình biến (loại, giá trị mặc định, mô tả, bắt buộc)
            
        Returns:
            Giá trị biến môi trường đã chuyển đổi
            
        Raises:
            ValueError: Nếu biến bắt buộc không được đặt
        """
        # Lấy giá trị từ môi trường hoặc sử dụng mặc định
        value = os.getenv(var_name, config.get("default", ""))
        
        # Kiểm tra xem biến có bắt buộc không
        if config.get("required", False) and not value:
            # Không raise lỗi mà chỉ in cảnh báo để tránh làm crash ứng dụng
            print(f"CẢNH BÁO: Biến môi trường bắt buộc '{var_name}' chưa được đặt!", file=sys.stderr)
        
        # Chuyển đổi giá trị sang đúng kiểu dữ liệu
        type_name = config.get("type", "str")
        type_converter = ENV_TYPES.get(type_name, str)
        
        try:
            # Chỉ chuyển đổi nếu có giá trị
            if value:
                return type_converter(value)
            return value
        except Exception as e:
            print(f"Lỗi khi chuyển đổi biến '{var_name}': {str(e)}", file=sys.stderr)
            # Trả về giá trị mặc định nếu có lỗi
            return config.get("default", "")
    
    def get(self, var_name: str, default: Any = None) -> Any:
        """
        Lấy giá trị biến môi trường.
        
        Args:
            var_name: Tên biến môi trường
            default: Giá trị mặc định nếu không tìm thấy
            
        Returns:
            Giá trị biến môi trường
        """
        return self.env_values.get(var_name, default)
    
    def set(self, var_name: str, value: Any) -> None:
        """
        Đặt giá trị biến môi trường.
        
        Args:
            var_name: Tên biến môi trường
            value: Giá trị cần đặt
        """
        self.env_values[var_name] = value
        # Cũng đặt biến môi trường của hệ thống
        os.environ[var_name] = str(value)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Lấy tất cả các biến môi trường.
        
        Returns:
            Dict các biến môi trường
        """
        return self.env_values.copy()
    
    def generate_env_file(self, output_path: Optional[str] = None) -> None:
        """
        Tạo file .env từ cấu hình hiện tại.
        
        Args:
            output_path: Đường dẫn file .env đầu ra
        """
        if output_path is None:
            output_path = BASE_DIR / ".env.example"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Automated Trading System - Environment Variables\n\n")
            
            # Nhóm biến theo category
            categories = {
                "# System": ["TRADING_ENV", "DEBUG_MODE", "LOG_LEVEL", "DATABASE_URI"],
                "# API & Security": ["API_PORT", "API_HOST", "SECRET_KEY", "TOKEN_EXPIRY", "SECURITY_LEVEL"],
                "# Exchanges": ["DEFAULT_EXCHANGE", "BINANCE_API_KEY", "BINANCE_API_SECRET", "BYBIT_API_KEY", "BYBIT_API_SECRET"],
                "# Trading": ["MAX_OPEN_POSITIONS", "DEFAULT_LEVERAGE", "RISK_PER_TRADE", "ASSET_LIST"],
                "# Training": ["RANDOM_SEED", "BATCH_SIZE", "LEARNING_RATE", "GAMMA", "EPSILON", "TRAIN_EPISODES"],
                "# Performance": ["MAX_THREADS", "MAX_PROCESSES", "REQUEST_TIMEOUT", "MAX_RETRIES", "MEMORY_LIMIT"]
            }
            
            for category, vars in categories.items():
                f.write(f"{category}\n")
                
                for var in vars:
                    if var in ENV_VARS:
                        config = ENV_VARS[var]
                        default = config.get("default", "")
                        description = config.get("description", "")
                        
                        f.write(f"# {description}\n")
                        if config.get("required", False):
                            f.write(f"{var}=\n\n")
                        else:
                            f.write(f"{var}={default}\n\n")
                
                f.write("\n")
        
        print(f"Đã tạo file .env mẫu tại {output_path}")
    
    def validate_required_vars(self) -> List[str]:
        """
        Kiểm tra xem các biến bắt buộc đã được đặt chưa.
        
        Returns:
            Danh sách các biến bắt buộc chưa được đặt
        """
        missing_vars = []
        
        for var_name, config in ENV_VARS.items():
            if config.get("required", False) and not self.get(var_name):
                missing_vars.append(var_name)
        
        return missing_vars

# Tạo instance mặc định
env_manager = EnvManager()

def get_env_manager() -> EnvManager:
    """Hàm helper để lấy instance EnvManager."""
    return env_manager

def get_env(var_name: str, default: Any = None) -> Any:
    """
    Hàm helper để lấy giá trị biến môi trường.
    
    Args:
        var_name: Tên biến môi trường
        default: Giá trị mặc định
        
    Returns:
        Giá trị biến môi trường
    """
    return env_manager.get(var_name, default)

def set_env(var_name: str, value: Any) -> None:
    """
    Hàm helper để đặt giá trị biến môi trường.
    
    Args:
        var_name: Tên biến môi trường
        value: Giá trị cần đặt
    """
    env_manager.set(var_name, value)

# Tạo file .env mẫu khi chạy trực tiếp file này
if __name__ == "__main__":
    env_manager.generate_env_file()