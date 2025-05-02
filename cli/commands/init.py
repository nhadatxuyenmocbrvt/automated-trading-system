"""
Module xử lý các lệnh từ dòng lệnh.
"""

from ..parser import create_parser
from .collect_commands import handle_collect
from .process_commands import handle_process
from .backtest_commands import handle_backtest
from .train_commands import handle_train
from .trade_commands import handle_trade
from .dashboard_commands import handle_dashboard

async def process_command(system, args):
    """
    Xử lý lệnh từ tham số dòng lệnh.
    
    Args:
        system: Hệ thống giao dịch tự động
        args: Tham số dòng lệnh
    """
    command = args.command
    
    if command == 'collect':
        await handle_collect(system, args)
    elif command == 'process':
        await handle_process(system, args)
    elif command == 'backtest':
        await handle_backtest(system, args)
    elif command == 'train':
        await handle_train(system, args)
    elif command == 'trade':
        await handle_trade(system, args)
    elif command == 'dashboard':
        await handle_dashboard(system, args)
    