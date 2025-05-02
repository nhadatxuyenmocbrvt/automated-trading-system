#!/usr/bin/env python3
"""
Điểm khởi chạy chính cho hệ thống giao dịch tự động.
File này cung cấp giao diện dòng lệnh đơn giản để sử dụng các chức năng
chính của hệ thống.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import argparse

# Thêm thư mục gốc vào PATH để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các module cần thiết
from cli.parser import create_parser
from cli.commands import collect_commands, process_commands, backtest_commands, train_commands, trade_commands, dashboard_commands
from config.logging_config import setup_logger
from trading_system import AutomatedTradingSystem

# Thiết lập logger
logger = setup_logger("main")

def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Xử lý exception không được bắt.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Cho phép Ctrl+C thoát bình thường
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Lỗi không được xử lý:", exc_info=(exc_type, exc_value, exc_traceback))

# Đặt handler cho các exception không được bắt
sys.excepthook = handle_exception

async def main():
    """
    Hàm main chính của ứng dụng.
    """
    # Tạo parser lệnh
    parser = create_parser()
    args = parser.parse_args()

    # Kiểm tra nếu không có lệnh nào được chỉ định
    if not hasattr(args, 'command'):
        parser.print_help()
        return
    
    # Tạo hệ thống giao dịch tự động
    trading_system = AutomatedTradingSystem(
        mode=args.mode,
        verbose=args.verbose
    )
    
    # Khởi tạo hệ thống
    await trading_system.setup()
    
    try:
        # Xử lý các lệnh
        if args.command == 'collect':
            await collect_commands.handle_collect_command(args, trading_system)
        elif args.command == 'process':
            await process_commands.handle_process_command(args, trading_system)
        elif args.command == 'backtest':
            await backtest_commands.handle_backtest_command(args, trading_system)
        elif args.command == 'train':
            await train_commands.handle_train_command(args, trading_system)
        elif args.command == 'trade':
            await trade_commands.handle_trade_command(args, trading_system)
        elif args.command == 'dashboard':
            await dashboard_commands.handle_dashboard_command(args, trading_system)
        else:
            logger.error(f"Lệnh không được hỗ trợ: {args.command}")
            
    except KeyboardInterrupt:
        logger.info("Đã nhận lệnh thoát từ người dùng (Ctrl+C)")
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện lệnh {args.command}: {str(e)}")
    finally:
        # Lưu trạng thái hệ thống
        if args.save_state:
            trading_system.save_system_state()

if __name__ == "__main__":
    try:
        # Chạy hàm main bất đồng bộ
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nChương trình đã bị dừng bởi người dùng")
        sys.exit(0)