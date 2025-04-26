"""
Test runner for the exchange connector tests.
"""

import os
import sys
import pytest

# Thêm thư mục hiện tại vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

if __name__ == "__main__":
    # Chạy tests với output chi tiết
    # Sử dụng đường dẫn tuyệt đối đến các file test
    test_files = [
        os.path.join(current_dir, "test_generic_connector.py"),
        os.path.join(current_dir, "test_binance_connector.py"),
        os.path.join(current_dir, "test_bybit_connector.py")
    ]
    
    # Chỉ chạy các file tồn tại
    existing_test_files = [f for f in test_files if os.path.exists(f)]
    
    if not existing_test_files:
        print("Không tìm thấy file test nào!")
        print(f"Đang tìm kiếm trong thư mục: {current_dir}")
        print("Các file hiện có trong thư mục:")
        print("\n".join(os.listdir(current_dir)))
        sys.exit(1)
        
    print(f"Đang chạy các file test: {existing_test_files}")
    pytest.main(["-v", "--no-header"] + existing_test_files)