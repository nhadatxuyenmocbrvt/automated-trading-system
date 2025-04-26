"""
Cấu hình bảo mật hệ thống.
File này cung cấp các cấu hình liên quan đến bảo mật, quản lý API key,
xác thực và các biện pháp bảo vệ khác.
"""

import os
import json
import uuid
import hashlib
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

from config.system_config import BASE_DIR, CONFIG_DIR, get_system_config
from config.utils.encryption import encrypt_data, decrypt_data, generate_key
from config.logging_config import get_logger

# Lấy cấu hình hệ thống và logger
system_config = get_system_config()
logger = get_logger("security")

# Đường dẫn lưu trữ API keys (đã mã hóa)
API_KEYS_FILE = CONFIG_DIR / "secure" / "api_keys.enc"
CONFIG_DIR.joinpath("secure").mkdir(exist_ok=True, parents=True)

# Thiết lập mặc định
DEFAULT_SECURITY_CONFIG = {
    # Cấu hình chung
    "security_level": os.getenv("SECURITY_LEVEL", "high"),  # low, medium, high
    "enable_ip_whitelisting": os.getenv("ENABLE_IP_WHITELISTING", "False").lower() == "true",
    "enable_rate_limiting": os.getenv("ENABLE_RATE_LIMITING", "True").lower() == "true",
    "session_timeout": int(os.getenv("SESSION_TIMEOUT", "3600")),  # seconds
    
    # Rate limiting
    "rate_limits": {
        "api": {
            "requests_per_minute": int(os.getenv("API_REQUESTS_PER_MINUTE", "60")),
            "requests_per_hour": int(os.getenv("API_REQUESTS_PER_HOUR", "1000")),
        },
        "exchange_api": {
            "requests_per_minute": int(os.getenv("EXCHANGE_REQUESTS_PER_MINUTE", "30")),
            "requests_per_hour": int(os.getenv("EXCHANGE_REQUESTS_PER_HOUR", "500")),
        },
    },
    
    # Danh sách IP được phép (nếu bật IP whitelisting)
    "ip_whitelist": os.getenv("IP_WHITELIST", "127.0.0.1").split(","),
    
    # Thiết lập xác thực
    "authentication": {
        "require_api_key": os.getenv("REQUIRE_API_KEY", "True").lower() == "true",
        "api_key_expiry_days": int(os.getenv("API_KEY_EXPIRY_DAYS", "90")),
        "password_min_length": int(os.getenv("PASSWORD_MIN_LENGTH", "12")),
        "password_require_special_char": os.getenv("PASSWORD_REQUIRE_SPECIAL", "True").lower() == "true",
        "password_require_number": os.getenv("PASSWORD_REQUIRE_NUMBER", "True").lower() == "true",
        "password_require_uppercase": os.getenv("PASSWORD_REQUIRE_UPPERCASE", "True").lower() == "true",
        "max_login_attempts": int(os.getenv("MAX_LOGIN_ATTEMPTS", "5")),
    },
    
    # Cấu hình SSL/TLS
    "ssl": {
        "enabled": os.getenv("SSL_ENABLED", "True").lower() == "true",
        "cert_path": os.getenv("SSL_CERT_PATH", str(CONFIG_DIR / "secure" / "cert.pem")),
        "key_path": os.getenv("SSL_KEY_PATH", str(CONFIG_DIR / "secure" / "key.pem")),
    },
    
    # Signing key cho JWT hoặc các token khác
    "signing_key": os.getenv("SIGNING_KEY", str(uuid.uuid4())),
}

class SecurityConfig:
    """
    Lớp quản lý cấu hình bảo mật.
    Cung cấp phương thức truy cập và quản lý cấu hình bảo mật, API keys.
    """
    
    def __init__(self):
        self.config = DEFAULT_SECURITY_CONFIG.copy()
        self._api_keys = {}
        self._master_key = self._get_or_create_master_key()
        
        # Tải API keys nếu tệp tồn tại
        if API_KEYS_FILE.exists():
            self._load_api_keys()
    
    def _get_or_create_master_key(self) -> bytes:
        """
        Lấy hoặc tạo khóa chính cho việc mã hóa/giải mã.
        
        Returns:
            Khóa mã hóa dưới dạng bytes
        """
        key_file = CONFIG_DIR / "secure" / "master.key"
        
        if key_file.exists():
            with open(key_file, "rb") as f:
                return base64.b64decode(f.read())
        else:
            # Tạo khóa mới
            key = generate_key()
            key_file.parent.mkdir(exist_ok=True, parents=True)
            with open(key_file, "wb") as f:
                f.write(base64.b64encode(key))
            
            logger.info("Đã tạo khóa chính mới cho bảo mật hệ thống")
            return key
    
    def _load_api_keys(self) -> None:
        """Tải danh sách API keys đã được mã hóa."""
        try:
            with open(API_KEYS_FILE, "rb") as f:
                encrypted_data = f.read()
            
            decrypted_data = decrypt_data(encrypted_data, self._master_key)
            self._api_keys = json.loads(decrypted_data.decode("utf-8"))
            logger.debug("Đã tải danh sách API keys")
        except Exception as e:
            logger.error(f"Lỗi khi tải API keys: {str(e)}")
            self._api_keys = {}
    
    def _save_api_keys(self) -> None:
        """Lưu danh sách API keys với mã hóa."""
        try:
            encrypted_data = encrypt_data(
                json.dumps(self._api_keys, ensure_ascii=False).encode("utf-8"),
                self._master_key
            )
            
            with open(API_KEYS_FILE, "wb") as f:
                f.write(encrypted_data)
            
            logger.debug("Đã lưu danh sách API keys")
        except Exception as e:
            logger.error(f"Lỗi khi lưu API keys: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Lấy giá trị cấu hình bảo mật.
        
        Args:
            key: Khóa cấu hình
            default: Giá trị mặc định nếu không tìm thấy
            
        Returns:
            Giá trị cấu hình
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Cập nhật giá trị cấu hình bảo mật.
        
        Args:
            key: Khóa cấu hình
            value: Giá trị cần đặt
        """
        keys = key.split('.')
        config = self.config
        
        # Duyệt đến cấp trước cấp cuối
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def add_api_key(self, exchange: str, api_key: str, api_secret: str, 
                    description: str = "", permissions: List[str] = None) -> str:
        """
        Thêm API key mới.
        
        Args:
            exchange: Tên sàn giao dịch
            api_key: API key
            api_secret: API secret
            description: Mô tả về key
            permissions: Danh sách quyền (read, trade, withdraw...)
            
        Returns:
            ID của key
        """
        if permissions is None:
            permissions = ["read"]
        
        key_id = str(uuid.uuid4())
        expiry_days = self.config["authentication"]["api_key_expiry_days"]
        expiry_date = (datetime.now() + timedelta(days=expiry_days)).isoformat()
        
        # Hash key để kiểm tra
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Lưu thông tin key
        self._api_keys[key_id] = {
            "exchange": exchange,
            "key_hash": key_hash,
            "api_key": api_key,  # Lưu ý: Trong môi trường thực tế, cần mã hóa thêm
            "api_secret": api_secret,  # Lưu ý: Trong môi trường thực tế, cần mã hóa thêm
            "description": description,
            "permissions": permissions,
            "created_at": datetime.now().isoformat(),
            "expires_at": expiry_date,
            "is_active": True
        }
        
        # Lưu danh sách keys
        self._save_api_keys()
        
        logger.info(f"Đã thêm API key mới cho sàn {exchange} với ID: {key_id}")
        return key_id
    
    def get_api_key(self, key_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin API key theo ID.
        
        Args:
            key_id: ID của key
            
        Returns:
            Thông tin key hoặc None nếu không tìm thấy
        """
        return self._api_keys.get(key_id)
    
    def get_exchange_keys(self, exchange: str) -> List[Dict[str, Any]]:
        """
        Lấy danh sách API keys cho một sàn giao dịch.
        
        Args:
            exchange: Tên sàn giao dịch
            
        Returns:
            Danh sách thông tin keys
        """
        return [
            {"id": key_id, **key_info}
            for key_id, key_info in self._api_keys.items()
            if key_info["exchange"].lower() == exchange.lower() and key_info["is_active"]
        ]
    
    def deactivate_api_key(self, key_id: str) -> bool:
        """
        Vô hiệu hóa API key.
        
        Args:
            key_id: ID của key
            
        Returns:
            True nếu thành công, False nếu không tìm thấy
        """
        if key_id in self._api_keys:
            self._api_keys[key_id]["is_active"] = False
            self._save_api_keys()
            logger.info(f"Đã vô hiệu hóa API key: {key_id}")
            return True
        return False
    
    def remove_api_key(self, key_id: str) -> bool:
        """
        Xóa API key khỏi hệ thống.
        
        Args:
            key_id: ID của key
            
        Returns:
            True nếu thành công, False nếu không tìm thấy
        """
        if key_id in self._api_keys:
            del self._api_keys[key_id]
            self._save_api_keys()
            logger.info(f"Đã xóa API key: {key_id}")
            return True
        return False
    
    def verify_api_key(self, api_key: str) -> bool:
        """
        Kiểm tra tính hợp lệ của API key.
        
        Args:
            api_key: API key cần kiểm tra
            
        Returns:
            True nếu key hợp lệ và đang hoạt động
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        for key_info in self._api_keys.values():
            if (key_info["key_hash"] == key_hash and 
                key_info["is_active"] and 
                datetime.fromisoformat(key_info["expires_at"]) > datetime.now()):
                return True
        
        return False
    
    def is_ip_allowed(self, ip_address: str) -> bool:
        """
        Kiểm tra xem địa chỉ IP có được phép truy cập không.
        
        Args:
            ip_address: Địa chỉ IP cần kiểm tra
            
        Returns:
            True nếu IP được phép hoặc không bật tính năng kiểm tra IP
        """
        if not self.config["enable_ip_whitelisting"]:
            return True
            
        return ip_address in self.config["ip_whitelist"]
    
    def add_to_ip_whitelist(self, ip_address: str) -> None:
        """
        Thêm địa chỉ IP vào whitelist.
        
        Args:
            ip_address: Địa chỉ IP cần thêm
        """
        if ip_address not in self.config["ip_whitelist"]:
            self.config["ip_whitelist"].append(ip_address)
            logger.info(f"Đã thêm IP {ip_address} vào whitelist")
    
    def remove_from_ip_whitelist(self, ip_address: str) -> bool:
        """
        Xóa địa chỉ IP khỏi whitelist.
        
        Args:
            ip_address: Địa chỉ IP cần xóa
            
        Returns:
            True nếu thành công, False nếu không tìm thấy
        """
        if ip_address in self.config["ip_whitelist"]:
            self.config["ip_whitelist"].remove(ip_address)
            logger.info(f"Đã xóa IP {ip_address} khỏi whitelist")
            return True
        return False
    
    def check_rate_limit(self, category: str, identifier: str) -> bool:
        """
        Kiểm tra giới hạn tốc độ.
        Có thể cài đặt thêm một bộ đếm thời gian thực hoặc sử dụng Redis.
        
        Args:
            category: Loại request (api, exchange_api)
            identifier: Định danh của người gửi (IP, user ID)
            
        Returns:
            True nếu chưa vượt quá giới hạn
        """
        # Đây là phiên bản đơn giản, cần cài đặt thêm bộ đếm thực tế
        if not self.config["enable_rate_limiting"]:
            return True
            
        # Giả định luôn trong giới hạn
        return True
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Lấy toàn bộ cấu hình bảo mật (đã ẩn thông tin nhạy cảm).
        
        Returns:
            Dictionary cấu hình
        """
        # Tạo bản sao để không làm thay đổi cấu hình gốc
        config_copy = self.config.copy()
        
        # Ẩn thông tin nhạy cảm
        if "signing_key" in config_copy:
            config_copy["signing_key"] = "********"
        
        return config_copy

# Tạo instance mặc định để sử dụng trong ứng dụng
security_config = SecurityConfig()

def get_security_config() -> SecurityConfig:
    """Hàm helper để lấy instance SecurityConfig."""
    return security_config