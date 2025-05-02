"""
Module commands.
Chứa các module xử lý lệnh dòng lệnh cho hệ thống giao dịch tự động.
"""

from cli.commands.collect_commands import setup_collect_parser, handle_collect_command
from cli.commands.process_commands import setup_process_parser, handle_process_command
from cli.commands.backtest_commands import setup_backtest_parser, handle_backtest_command
from cli.commands.train_commands import setup_train_parser, handle_train_command
from cli.commands.trade_commands import setup_trade_parser, handle_trade_command
from cli.commands.dashboard_commands import setup_dashboard_parser, handle_dashboard_command