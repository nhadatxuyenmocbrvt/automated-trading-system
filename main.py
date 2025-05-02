#!/usr/bin/env python3
"""
Điểm khởi chạy chính của Hệ thống Giao dịch Tự động.
File này cung cấp giao diện dòng lệnh để khởi chạy các thành phần
khác nhau của hệ thống, bao gồm thu thập dữ liệu, xử lý dữ liệu, huấn luyện agent,
backtest, và triển khai giao dịch thực tế.
"""

import asyncio
import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import các module
sys.path.append(str(Path(__file__).parent))

from config.logging_config import setup_component_logger
from cli.parser import create_parser
from trading_system import AutomatedTradingSystem

# Thiết lập logger cho main
logger = setup_component_logger("main", "INFO")

async def main():
    """
    Hàm main chính của hệ thống.
    """
    # Tạo parser và parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Nếu không có lệnh nào được cung cấp, hiển thị help
    if not hasattr(args, 'command') or not args.command:
        parser.print_help()
        return
    
    # Khởi tạo hệ thống
    system = AutomatedTradingSystem()
    
    try:
        # Xử lý lệnh từ cli.commands
        from cli.commands import process_command
        await process_command(system, args)
        
    except KeyboardInterrupt:
        logger.info("Nhận tín hiệu dừng từ người dùng")
    except Exception as e:
        logger.error(f"Lỗi không xử lý được: {str(e)}", exc_info=True)
        
        # Đưa ra gợi ý giải pháp dựa trên loại lỗi
        if "timeout" in str(e).lower():
            logger.warning("Lỗi timeout có thể do kết nối mạng không ổn định hoặc sàn giao dịch không phản hồi")
            logger.warning("Giải pháp: Tăng REQUEST_TIMEOUT trong .env, kiểm tra kết nối mạng hoặc sử dụng proxy")
        elif "headers" in str(e).lower():
            logger.warning("Lỗi xử lý response từ sàn giao dịch. Có thể do cấu trúc response không đúng định dạng.")
            logger.warning("Giải pháp: Cập nhật thư viện CCXT hoặc kiểm tra lại các connector")
    finally:
        # Dọn dẹp tài nguyên
        await system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())