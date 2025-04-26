"""
Tiện ích mã hóa dữ liệu.
File này cung cấp các hàm mã hóa/giải mã dữ liệu nhạy cảm,
bao gồm API keys, mật khẩu và thông tin xác thực khác.
"""

import os
import base64
import hashlib
from typing import Union, Tuple, Optional

# Sử dụng thư viện mã hóa an toàn
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

def generate_key() -> bytes:
    """
    Tạo khóa mã hóa Fernet ngẫu nhiên.
    
    Returns:
        Khóa mã hóa dưới dạng bytes
    """
    return Fernet.generate_key()

def derive_key(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """
    Tạo khóa mã hóa từ mật khẩu và salt.
    
    Args:
        password: Mật khẩu làm seed
        salt: Salt ngẫu nhiên (tự tạo nếu không cung cấp)
        
    Returns:
        Tuple (key, salt)
    """
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key, salt

def encrypt_data(data: Union[str, bytes], key: bytes) -> bytes:
    """
    Mã hóa dữ liệu sử dụng Fernet.
    
    Args:
        data: Dữ liệu cần mã hóa (string hoặc bytes)
        key: Khóa mã hóa (dạng bytes)
        
    Returns:
        Dữ liệu đã mã hóa (bytes)
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    f = Fernet(key)
    return f.encrypt(data)

def decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
    """
    Giải mã dữ liệu đã được mã hóa bằng Fernet.
    
    Args:
        encrypted_data: Dữ liệu đã mã hóa (bytes)
        key: Khóa mã hóa (bytes)
        
    Returns:
        Dữ liệu đã giải mã (bytes)
        
    Raises:
        cryptography.fernet.InvalidToken: Nếu mã thông báo không hợp lệ hoặc sai khóa
    """
    f = Fernet(key)
    return f.decrypt(encrypted_data)

def hash_password(password: str) -> Tuple[str, str]:
    """
    Băm mật khẩu với salt ngẫu nhiên.
    
    Args:
        password: Mật khẩu cần băm
        
    Returns:
        Tuple (password_hash, salt) dưới dạng chuỗi hex
    """
    salt = os.urandom(32)
    
    # Tạo hash với salt
    password_hash = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'),
        salt,
        100000
    )
    
    # Chuyển sang hex để lưu trữ
    return password_hash.hex(), salt.hex()

def verify_password(stored_password_hash: str, stored_salt: str, provided_password: str) -> bool:
    """
    Xác minh mật khẩu với hash đã lưu.
    
    Args:
        stored_password_hash: Hash mật khẩu đã lưu (chuỗi hex)
        stored_salt: Salt đã lưu (chuỗi hex)
        provided_password: Mật khẩu được cung cấp để kiểm tra
        
    Returns:
        True nếu mật khẩu khớp, False nếu không
    """
    # Chuyển salt từ hex sang bytes
    salt = bytes.fromhex(stored_salt)
    
    # Tạo hash từ mật khẩu được cung cấp
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        provided_password.encode('utf-8'),
        salt,
        100000
    )
    
    # So sánh hash
    return password_hash.hex() == stored_password_hash

def encrypt_config_value(value: str, key: bytes) -> str:
    """
    Mã hóa một giá trị cấu hình để lưu trữ.
    
    Args:
        value: Giá trị cần mã hóa
        key: Khóa mã hóa
        
    Returns:
        Chuỗi mã hóa base64
    """
    encrypted = encrypt_data(value, key)
    return base64.b64encode(encrypted).decode('utf-8')

def decrypt_config_value(encrypted_value: str, key: bytes) -> str:
    """
    Giải mã một giá trị cấu hình đã mã hóa.
    
    Args:
        encrypted_value: Giá trị đã mã hóa (chuỗi base64)
        key: Khóa mã hóa
        
    Returns:
        Giá trị đã giải mã
    """
    encrypted_bytes = base64.b64decode(encrypted_value)
    decrypted_bytes = decrypt_data(encrypted_bytes, key)
    return decrypted_bytes.decode('utf-8')

def create_api_key_hash(api_key: str) -> str:
    """
    Tạo hash từ API key để lưu trữ an toàn.
    
    Args:
        api_key: API key cần hash
        
    Returns:
        Hash của API key
    """
    return hashlib.sha256(api_key.encode()).hexdigest()

def secure_compare(a: str, b: str) -> bool:
    """
    So sánh hai chuỗi theo thời gian không đổi.
    Giúp tránh timing attacks.
    
    Args:
        a: Chuỗi thứ nhất
        b: Chuỗi thứ hai
        
    Returns:
        True nếu hai chuỗi giống nhau, False nếu không
    """
    if len(a) != len(b):
        return False
    
    result = 0
    for x, y in zip(a, b):
        result |= ord(x) ^ ord(y)
    
    return result == 0

def generate_random_token(length: int = 32) -> str:
    """
    Tạo token ngẫu nhiên an toàn.
    
    Args:
        length: Độ dài token (số bytes)
        
    Returns:
        Token ngẫu nhiên dưới dạng chuỗi hex
    """
    return os.urandom(length).hex()

def encrypt_api_credentials(api_key: str, api_secret: str, master_key: bytes) -> Tuple[str, str]:
    """
    Mã hóa thông tin đăng nhập API.
    
    Args:
        api_key: API key
        api_secret: API secret
        master_key: Khóa chính để mã hóa
        
    Returns:
        Tuple (encrypted_key, encrypted_secret) dưới dạng chuỗi base64
    """
    encrypted_key = encrypt_config_value(api_key, master_key)
    encrypted_secret = encrypt_config_value(api_secret, master_key)
    
    return encrypted_key, encrypted_secret

def decrypt_api_credentials(encrypted_key: str, encrypted_secret: str, master_key: bytes) -> Tuple[str, str]:
    """
    Giải mã thông tin đăng nhập API.
    
    Args:
        encrypted_key: API key đã mã hóa (chuỗi base64)
        encrypted_secret: API secret đã mã hóa (chuỗi base64)
        master_key: Khóa chính để giải mã
        
    Returns:
        Tuple (api_key, api_secret)
    """
    api_key = decrypt_config_value(encrypted_key, master_key)
    api_secret = decrypt_config_value(encrypted_secret, master_key)
    
    return api_key, api_secret