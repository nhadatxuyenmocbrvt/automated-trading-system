"""
Parser dòng lệnh chính.
File này xây dựng bộ parser lệnh chính và các subparsers cho các chức năng khác nhau.
"""

import argparse
import os
import sys
from typing import Dict, List, Any, Callable
from pathlib import Path

# Thêm thư mục gốc vào path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các module xử lý lệnh
from cli.commands.collect_commands import setup_collect_parser, handle_collect_command
from cli.commands.process_commands import setup_process_parser, handle_process_command
from cli.commands.backtest_commands import setup_backtest_parser, handle_backtest_command
from cli.commands.train_commands import setup_train_parser, handle_train_command
from cli.commands.trade_commands import setup_trade_parser, handle_trade_command
from cli.commands.dashboard_commands import setup_dashboard_parser, handle_dashboard_command

def create_parser() -> argparse.ArgumentParser:
    """
    Tạo parser chính cho CLI với các subparser.
    
    Returns:
        ArgumentParser: Parser dòng lệnh đã cấu hình
    """
    # Tạo parser chính
    parser = argparse.ArgumentParser(
        description='Hệ thống giao dịch tự động sử dụng Reinforcement Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Thêm các arguments chung
    parser.add_argument('--debug', action='store_true', help='Bật chế độ debug')
    parser.add_argument('--config', type=str, help='Đường dẫn file cấu hình')
    
    # Tạo subparsers
    subparsers = parser.add_subparsers(title='commands', dest='command')
    
    # Thiết lập các subparser
    setup_collect_parser(subparsers)
    setup_process_parser(subparsers)
    setup_backtest_parser(subparsers)
    setup_train_parser(subparsers)
    setup_trade_parser(subparsers)
    setup_dashboard_parser(subparsers)
    
    return parser

def register_command_handler(parser: argparse.ArgumentParser, subcommand: str, handler_func: Callable) -> None:
    """
    Đăng ký hàm xử lý cho một subcommand.
    
    Args:
        parser: Parser chính
        subcommand: Tên subcommand
        handler_func: Hàm xử lý
    """
    if hasattr(parser, '_subparsers'):
        for action in parser._subparsers._actions:
            if isinstance(action, argparse._SubParsersAction):
                for choice, subparser in action.choices.items():
                    if choice == subcommand:
                        subparser.set_defaults(func=handler_func)
                        return
    
    raise ValueError(f"Không tìm thấy subcommand '{subcommand}'")

if __name__ == "__main__":
    # Test parser
    parser = create_parser()
    args = parser.parse_args()
    print(args)