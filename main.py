"""
Điểm khởi chạy chính của hệ thống.
File này cung cấp giao diện dòng lệnh (CLI) cho người dùng,
cho phép điều khiển các chức năng chính của hệ thống giao dịch tự động.
"""

import os
import sys
import logging
from pathlib import Path

# Thêm thư mục gốc vào path để import các module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import các module cần thiết
from config.logging_config import get_logger
from config.system_config import get_system_config
from cli.parser import create_parser
from trading_system import AutomatedTradingSystem

def main():
    """
    Hàm chính điều khiển luồng thực thi của ứng dụng.
    """
    # Khởi tạo logger
    logger = get_logger("main")
    logger.info("Khởi động hệ thống giao dịch tự động")
    
    # Lấy cấu hình hệ thống
    system_config = get_system_config()
    
    # Tạo parser dòng lệnh
    parser = create_parser()
    args = parser.parse_args()
    
    # Khởi tạo hệ thống giao dịch
    trading_system = AutomatedTradingSystem(config=system_config)
    
    # Nếu không có lệnh nào được cung cấp, hiển thị help
    if not hasattr(args, 'func'):
        parser.print_help()
        return
    
    try:
        # Chạy lệnh được chọn
        args.func(args, trading_system)
    except KeyboardInterrupt:
        logger.info("Nhận tín hiệu ngắt. Đang dừng hệ thống...")
    except Exception as e:
        logger.error(f"Lỗi không mong đợi: {str(e)}", exc_info=True)
        return 1
    finally:
        logger.info("Đóng hệ thống giao dịch tự động")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())