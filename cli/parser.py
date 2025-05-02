"""
Module xây dựng parser cho các lệnh dòng lệnh.
"""

import argparse

def create_parser() -> argparse.ArgumentParser:
    """
    Tạo và cấu hình parser cho các lệnh dòng lệnh.
    
    Returns:
        ArgumentParser đã cấu hình
    """
    # Tạo parser chính
    parser = argparse.ArgumentParser(description='Hệ thống Giao dịch Tự động')
    subparsers = parser.add_subparsers(dest='command', help='Lệnh cần thực hiện')
    
    # Thêm các subparser
    _add_collect_parser(subparsers)
    _add_process_parser(subparsers)
    _add_backtest_parser(subparsers)
    _add_train_parser(subparsers)
    _add_trade_parser(subparsers)
    _add_dashboard_parser(subparsers)
    
    return parser

def _add_collect_parser(subparsers):
    """Thêm subparser cho lệnh collect"""
    collect_parser = subparsers.add_parser('collect', help='Thu thập dữ liệu lịch sử')
    collect_parser.add_argument('--exchange', type=str, default='binance', help='Tên sàn giao dịch')
    collect_parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDT', 'ETH/USDT'], 
                               help='Danh sách cặp giao dịch')
    collect_parser.add_argument('--timeframes', type=str, nargs='+', default=['1h', '4h', '1d'], 
                               help='Danh sách timeframe')
    collect_parser.add_argument('--days', type=int, default=30, help='Số ngày lấy dữ liệu')
    collect_parser.add_argument('--futures', action='store_true', help='Sử dụng thị trường futures')

# [Thêm các hàm tương tự cho các subparser khác: _add_process_parser, _add_backtest_parser, vv.]