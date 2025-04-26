"""
Tiện ích xác thực dữ liệu.
File này cung cấp các hàm xác thực dữ liệu đầu vào,
cấu hình, và các tham số trong toàn bộ hệ thống.
"""

import re
import ipaddress
import json
from typing import Any, Dict, List, Union, Tuple, Optional, Callable
from datetime import datetime

class ValidationError(Exception):
    """Exception xảy ra khi xác thực dữ liệu thất bại."""
    pass

# --- Xác thực cấu hình chung ---

def validate_config(config: Dict[str, Any], schema: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Xác thực cấu hình dựa trên schema.
    
    Args:
        config: Từ điển cấu hình cần xác thực
        schema: Schema xác thực
        
    Returns:
        Danh sách các lỗi (rỗng nếu hợp lệ)
    """
    errors = []
    
    for key, value_schema in schema.items():
        # Kiểm tra trường bắt buộc
        if value_schema.get("required", False) and key not in config:
            errors.append(f"Thiếu trường bắt buộc: {key}")
            continue
            
        # Bỏ qua nếu không có trong config
        if key not in config:
            continue
            
        # Lấy giá trị từ config
        value = config[key]
        
        # Kiểm tra kiểu dữ liệu
        if "type" in value_schema:
            expected_type = value_schema["type"]
            if not isinstance(value, _get_python_type(expected_type)):
                errors.append(f"Kiểu dữ liệu không hợp lệ cho {key}: mong đợi {expected_type}, nhận được {type(value).__name__}")
        
        # Kiểm tra giá trị enum
        if "enum" in value_schema and value not in value_schema["enum"]:
            errors.append(f"Giá trị {value} không hợp lệ cho {key}, phải là một trong: {', '.join(map(str, value_schema['enum']))}")
        
        # Kiểm tra giới hạn
        if "min" in value_schema and value < value_schema["min"]:
            errors.append(f"Giá trị {value} cho {key} nhỏ hơn tối thiểu {value_schema['min']}")
            
        if "max" in value_schema and value > value_schema["max"]:
            errors.append(f"Giá trị {value} cho {key} lớn hơn tối đa {value_schema['max']}")
        
        # Kiểm tra mẫu
        if "pattern" in value_schema and isinstance(value, str):
            pattern = re.compile(value_schema["pattern"])
            if not pattern.match(value):
                errors.append(f"Giá trị {value} cho {key} không khớp với mẫu {value_schema['pattern']}")
        
        # Kiểm tra các giá trị lồng nhau
        if "properties" in value_schema and isinstance(value, dict):
            nested_errors = validate_config(value, value_schema["properties"])
            for error in nested_errors:
                errors.append(f"{key}.{error}")
    
    return errors

def _get_python_type(type_name: str) -> type:
    """Chuyển đổi tên kiểu dữ liệu thành lớp Python."""
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None)
    }
    return type_map.get(type_name, object)

# --- Xác thực chuỗi ---

def is_valid_string(value: Any, min_length: int = 0, max_length: Optional[int] = None, 
                    pattern: Optional[str] = None) -> bool:
    """
    Kiểm tra chuỗi hợp lệ.
    
    Args:
        value: Giá trị cần kiểm tra
        min_length: Độ dài tối thiểu
        max_length: Độ dài tối đa
        pattern: Mẫu regex để kiểm tra
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    if not isinstance(value, str):
        return False
        
    # Kiểm tra độ dài
    if len(value) < min_length:
        return False
        
    if max_length is not None and len(value) > max_length:
        return False
    
    # Kiểm tra mẫu
    if pattern is not None:
        return bool(re.match(pattern, value))
        
    return True

def is_valid_email(email: str) -> bool:
    """
    Kiểm tra địa chỉ email hợp lệ.
    
    Args:
        email: Địa chỉ email
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def is_valid_password(password: str, min_length: int = 8, 
                      require_uppercase: bool = True,
                      require_lowercase: bool = True,
                      require_digit: bool = True,
                      require_special: bool = True) -> Tuple[bool, str]:
    """
    Kiểm tra mật khẩu hợp lệ.
    
    Args:
        password: Mật khẩu cần kiểm tra
        min_length: Độ dài tối thiểu
        require_uppercase: Yêu cầu chữ hoa
        require_lowercase: Yêu cầu chữ thường
        require_digit: Yêu cầu chữ số
        require_special: Yêu cầu ký tự đặc biệt
        
    Returns:
        Tuple (is_valid, message)
    """
    if len(password) < min_length:
        return False, f"Mật khẩu phải có ít nhất {min_length} ký tự"
    
    if require_uppercase and not any(c.isupper() for c in password):
        return False, "Mật khẩu phải chứa ít nhất một chữ hoa"
    
    if require_lowercase and not any(c.islower() for c in password):
        return False, "Mật khẩu phải chứa ít nhất một chữ thường"
    
    if require_digit and not any(c.isdigit() for c in password):
        return False, "Mật khẩu phải chứa ít nhất một chữ số"
    
    if require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Mật khẩu phải chứa ít nhất một ký tự đặc biệt"
    
    return True, "Mật khẩu hợp lệ"

# --- Xác thực số ---

def is_valid_number(value: Any, min_value: Optional[float] = None, 
                   max_value: Optional[float] = None, 
                   is_integer: bool = False) -> bool:
    """
    Kiểm tra số hợp lệ.
    
    Args:
        value: Giá trị cần kiểm tra
        min_value: Giá trị tối thiểu
        max_value: Giá trị tối đa
        is_integer: True nếu yêu cầu số nguyên
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    # Kiểm tra kiểu dữ liệu
    if is_integer and not isinstance(value, int):
        return False
    
    if not isinstance(value, (int, float)):
        return False
    
    # Kiểm tra giá trị
    if min_value is not None and value < min_value:
        return False
        
    if max_value is not None and value > max_value:
        return False
        
    return True

def is_valid_percentage(value: Any) -> bool:
    """
    Kiểm tra giá trị phần trăm hợp lệ (0-100).
    
    Args:
        value: Giá trị cần kiểm tra
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    return is_valid_number(value, 0, 100)

# --- Xác thực danh sách ---

def is_valid_list(value: Any, item_validator: Optional[Callable[[Any], bool]] = None, 
                 min_length: int = 0, max_length: Optional[int] = None) -> bool:
    """
    Kiểm tra danh sách hợp lệ.
    
    Args:
        value: Giá trị cần kiểm tra
        item_validator: Hàm xác thực cho mỗi phần tử
        min_length: Độ dài tối thiểu
        max_length: Độ dài tối đa
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    if not isinstance(value, list):
        return False
        
    # Kiểm tra độ dài
    if len(value) < min_length:
        return False
        
    if max_length is not None and len(value) > max_length:
        return False
    
    # Kiểm tra các phần tử
    if item_validator is not None:
        return all(item_validator(item) for item in value)
        
    return True

# --- Xác thực định dạng ---

def is_valid_json(value: str) -> bool:
    """
    Kiểm tra chuỗi JSON hợp lệ.
    
    Args:
        value: Chuỗi cần kiểm tra
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    try:
        json.loads(value)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

def is_valid_ip_address(value: str) -> bool:
    """
    Kiểm tra địa chỉ IP hợp lệ.
    
    Args:
        value: Địa chỉ IP
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False

def is_valid_date(value: str, format: str = "%Y-%m-%d") -> bool:
    """
    Kiểm tra chuỗi ngày hợp lệ.
    
    Args:
        value: Chuỗi ngày
        format: Định dạng ngày
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    try:
        datetime.strptime(value, format)
        return True
    except ValueError:
        return False

def is_valid_url(value: str) -> bool:
    """
    Kiểm tra URL hợp lệ.
    
    Args:
        value: URL
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    pattern = r'^(https?):\/\/[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, value))

# --- Xác thực giao dịch ---

def is_valid_trading_pair(symbol: str) -> bool:
    """
    Kiểm tra cặp giao dịch hợp lệ.
    
    Args:
        symbol: Cặp giao dịch (ví dụ: BTC/USDT)
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    pattern = r'^[A-Z0-9]+/[A-Z0-9]+$'
    return bool(re.match(pattern, symbol))

def is_valid_timeframe(timeframe: str) -> bool:
    """
    Kiểm tra khung thời gian hợp lệ.
    
    Args:
        timeframe: Khung thời gian (1m, 5m, 1h, 1d, ...)
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    pattern = r'^[1-9][0-9]*(m|h|d|w|M)$'
    return bool(re.match(pattern, timeframe))

def is_valid_order_type(order_type: str) -> bool:
    """
    Kiểm tra loại lệnh hợp lệ.
    
    Args:
        order_type: Loại lệnh (market, limit, ...)
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    valid_types = ["market", "limit", "stop_loss", "take_profit", "stop_limit"]
    return order_type in valid_types

def is_valid_leverage(leverage: float) -> bool:
    """
    Kiểm tra đòn bẩy hợp lệ.
    
    Args:
        leverage: Đòn bẩy
        
    Returns:
        True nếu hợp lệ (1-125), False nếu không
    """
    return is_valid_number(leverage, 1, 125)

# --- Xác thực huấn luyện ---

def is_valid_learning_rate(learning_rate: float) -> bool:
    """
    Kiểm tra tốc độ học hợp lệ.
    
    Args:
        learning_rate: Tốc độ học
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    return is_valid_number(learning_rate, 0, 1)

def is_valid_batch_size(batch_size: int) -> bool:
    """
    Kiểm tra kích thước batch hợp lệ.
    
    Args:
        batch_size: Kích thước batch
        
    Returns:
        True nếu hợp lệ (phải là số dương và lũy thừa của 2), False nếu không
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        return False
    
    # Kiểm tra lũy thừa của 2
    return (batch_size & (batch_size - 1) == 0)

# --- Xác thực an toàn ---

def validate_input(value: Any, validators: List[Callable[[Any], Union[bool, Tuple[bool, str]]]]) -> Tuple[bool, str]:
    """
    Xác thực đầu vào với nhiều validator.
    
    Args:
        value: Giá trị cần xác thực
        validators: Danh sách các hàm xác thực
        
    Returns:
        Tuple (is_valid, message)
    """
    for validator in validators:
        result = validator(value)
        
        # Xử lý trường hợp validator trả về tuple
        if isinstance(result, tuple):
            is_valid, message = result
            if not is_valid:
                return False, message
        # Xử lý trường hợp validator trả về boolean
        elif not result:
            return False, "Giá trị không hợp lệ"
    
    return True, "Giá trị hợp lệ"

def sanitize_string(value: str) -> str:
    """
    Làm sạch chuỗi để tránh lỗi bảo mật.
    
    Args:
        value: Chuỗi cần làm sạch
        
    Returns:
        Chuỗi đã làm sạch
    """
    # Loại bỏ các ký tự điều khiển và không an toàn
    return re.sub(r'[^\w\s.,\-_@#%&*()[\]{}:;?!/+=]', '', value)

def is_safe_file_path(path: str) -> bool:
    """
    Kiểm tra đường dẫn file an toàn.
    
    Args:
        path: Đường dẫn cần kiểm tra
        
    Returns:
        True nếu an toàn, False nếu không
    """
    # Kiểm tra path traversal
    if '..' in path or '//' in path:
        return False
    
    # Kiểm tra ký tự không an toàn
    unsafe_chars = ['<', '>', '|', ':', '"', '?', '*']
    if any(c in path for c in unsafe_chars):
        return False
    
    return True