"""
Xử lý lệnh giao dịch thực tế.
File này định nghĩa các tham số và xử lý cho lệnh 'trade' trên CLI.
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# Thêm thư mục gốc vào path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import các module cần thiết
from config.logging_config import get_logger
from config.constants import OrderType, TimeInForce, OrderStatus, PositionSide, ErrorCode
from data_collectors.exchange_api.generic_connector import ExchangeConnector
from deployment.trade_executor import TradeExecutor
from deployment.exchange_api.order_manager import OrderManager
from deployment.exchange_api.account_manager import AccountManager
from deployment.exchange_api.position_tracker import PositionTracker
from risk_management.position_sizer import PositionSizer
from risk_management.stop_loss import StopLoss
from risk_management.take_profit import TakeProfit
from risk_management.risk_calculator import RiskCalculator
from trading_system import AutomatedTradingSystem

def setup_trade_parser(subparsers) -> None:
    """
    Thiết lập parser cho lệnh 'trade'.
    
    Args:
        subparsers: Subparsers object từ argparse
    """
    trade_parser = subparsers.add_parser(
        'trade',
        help='Bắt đầu giao dịch thực tế',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Tạo subcommands cho trade
    trade_subparsers = trade_parser.add_subparsers(dest='trade_command', help='Lệnh giao dịch')
    
    # Lệnh mở vị thế
    open_parser = trade_subparsers.add_parser('open', help='Mở vị thế mới')
    open_parser.add_argument('symbol', type=str, help='Cặp giao dịch (ví dụ: BTC/USDT)')
    open_parser.add_argument('side', choices=['long', 'short'], help='Hướng giao dịch')
    open_parser.add_argument('--size', type=float, required=True, help='Kích thước vị thế')
    open_parser.add_argument('--price', type=float, help='Giá vào lệnh (không bắt buộc cho lệnh market)')
    open_parser.add_argument('--type', type=str, choices=['market', 'limit'], default='market', help='Loại lệnh')
    open_parser.add_argument('--leverage', type=float, default=1.0, help='Đòn bẩy')
    open_parser.add_argument('--sl', type=float, help='Giá stop loss')
    open_parser.add_argument('--tp', type=float, help='Giá take profit')
    open_parser.add_argument('--trailing-stop', action='store_true', help='Sử dụng trailing stop')
    
    # Lệnh đóng vị thế
    close_parser = trade_subparsers.add_parser('close', help='Đóng vị thế')
    close_parser.add_argument('symbol', type=str, help='Cặp giao dịch (ví dụ: BTC/USDT)')
    close_parser.add_argument('--size', type=float, help='Kích thước cần đóng (không bắt buộc, mặc định là toàn bộ)')
    close_parser.add_argument('--price', type=float, help='Giá thoát (không bắt buộc)')
    
    # Lệnh đóng tất cả vị thế
    close_all_parser = trade_subparsers.add_parser('close-all', help='Đóng tất cả vị thế')
    close_all_parser.add_argument('--reason', type=str, default='manual', help='Lý do đóng vị thế')
    
    # Lệnh xem trạng thái
    status_parser = trade_subparsers.add_parser('status', help='Xem trạng thái vị thế/lệnh')
    status_parser.add_argument('--symbol', type=str, help='Cặp giao dịch (không bắt buộc)')
    status_parser.add_argument('--order-id', type=str, help='ID lệnh (không bắt buộc)')
    
    # Lệnh hủy lệnh
    cancel_parser = trade_subparsers.add_parser('cancel', help='Hủy lệnh')
    cancel_parser.add_argument('order_id', type=str, help='ID lệnh cần hủy')
    cancel_parser.add_argument('symbol', type=str, help='Cặp giao dịch')
    
    # Lệnh tạo báo cáo
    report_parser = trade_subparsers.add_parser('report', help='Tạo báo cáo hiệu suất')
    report_parser.add_argument('--days', type=int, default=30, help='Số ngày cần phân tích')
    report_parser.add_argument('--output', type=str, help='Đường dẫn file đầu ra (mặc định là console)')
    
    # Các tham số chung
    trade_parser.add_argument(
        '--exchange', 
        type=str, 
        default='binance',
        help='Sàn giao dịch (binance, bybit, ...)'
    )
    
    trade_parser.add_argument(
        '--api-key',
        type=str,
        help='API Key của sàn giao dịch'
    )
    
    trade_parser.add_argument(
        '--api-secret',
        type=str,
        help='API Secret của sàn giao dịch'
    )
    
    # Chạy ở chế độ paper trading
    trade_parser.add_argument(
        '--paper-trading', 
        action='store_true',
        help='Chạy ở chế độ paper trading (không giao dịch thực tế)'
    )
    
    trade_parser.add_argument(
        '--paper-balance', 
        type=float, 
        default=10000.0,
        help='Số dư ban đầu cho paper trading'
    )
    
    # Thiết lập quản lý rủi ro mặc định
    trade_parser.add_argument(
        '--max-positions', 
        type=int, 
        default=5,
        help='Số vị thế tối đa có thể mở'
    )
    
    trade_parser.add_argument(
        '--max-leverage', 
        type=float, 
        default=5.0,
        help='Đòn bẩy tối đa'
    )
    
    trade_parser.add_argument(
        '--default-sl-percent', 
        type=float, 
        default=0.05,
        help='Phần trăm stop loss mặc định (0.05 = 5%)'
    )
    
    trade_parser.add_argument(
        '--default-tp-percent', 
        type=float, 
        default=0.1,
        help='Phần trăm take profit mặc định (0.1 = 10%)'
    )
    
    trade_parser.set_defaults(func=handle_trade_command)

def handle_trade_command(args: argparse.Namespace, system: AutomatedTradingSystem) -> int:
    """
    Xử lý lệnh 'trade'.
    
    Args:
        args: Các tham số dòng lệnh
        system: Instance của AutomatedTradingSystem
        
    Returns:
        int: Mã kết quả (0 = thành công)
    """
    logger = get_logger('trade_command')
    
    # Khởi tạo kết nối đến sàn giao dịch
    try:
        # Tạo cấu hình cho exchange connector
        exchange_config = {
            'api_key': args.api_key,
            'api_secret': args.api_secret,
            'exchange_id': args.exchange
        }
        
        # Trường hợp paper trading
        if args.paper_trading:
            exchange_config['sandbox'] = True
            logger.info(f"Chế độ: Paper trading (Số dư: {args.paper_balance})")
        else:
            logger.info("Chế độ: Giao dịch thực tế")
        
        # Khởi tạo exchange connector
        exchange_connector = ExchangeConnector(**exchange_config)
        
        # Khởi tạo các thành phần quản lý giao dịch
        order_manager = OrderManager(exchange_connector=exchange_connector)
        account_manager = AccountManager(exchange_connector=exchange_connector)
        position_tracker = PositionTracker(exchange_connector=exchange_connector)
        
        # Khởi tạo các thành phần quản lý rủi ro
        position_sizer = PositionSizer()
        stop_loss_manager = StopLoss()
        take_profit_manager = TakeProfit()
        risk_calculator = RiskCalculator()
        
        # Khởi tạo TradeExecutor
        trade_executor = TradeExecutor(
            exchange_connector=exchange_connector,
            order_manager=order_manager,
            position_tracker=position_tracker,
            position_sizer=position_sizer,
            stop_loss_manager=stop_loss_manager,
            take_profit_manager=take_profit_manager,
            risk_calculator=risk_calculator,
            max_active_positions=args.max_positions,
            max_leverage=args.max_leverage,
            default_stop_loss_percent=args.default_sl_percent,
            default_take_profit_percent=args.default_tp_percent,
            dry_run=args.paper_trading
        )
        
        # Bắt đầu trade executor
        trade_executor.start()
        
        # Xử lý lệnh dựa trên trade_command
        if hasattr(args, 'trade_command'):
            if args.trade_command == 'open':
                # Mở vị thế mới
                result = open_position(args, trade_executor)
            elif args.trade_command == 'close':
                # Đóng vị thế
                result = close_position(args, trade_executor)
            elif args.trade_command == 'close-all':
                # Đóng tất cả vị thế
                result = close_all_positions(args, trade_executor)
            elif args.trade_command == 'status':
                # Xem trạng thái
                result = check_status(args, trade_executor, order_manager, account_manager)
            elif args.trade_command == 'cancel':
                # Hủy lệnh
                result = cancel_order(args, order_manager)
            elif args.trade_command == 'report':
                # Tạo báo cáo
                result = generate_report(args, trade_executor)
            else:
                logger.warning("Chưa chọn lệnh giao dịch cụ thể")
                return 1
        else:
            logger.warning("Chưa chọn lệnh giao dịch cụ thể")
            return 1
        
        # Dừng trade executor
        trade_executor.stop()
        
        # Hiển thị kết quả
        if isinstance(result, dict):
            logger.info(json.dumps(result, indent=2))
        
        return 0
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý lệnh giao dịch: {str(e)}")
        return 1

def open_position(args: argparse.Namespace, trade_executor: TradeExecutor) -> Dict[str, Any]:
    """
    Mở vị thế mới.
    
    Args:
        args: Các tham số dòng lệnh
        trade_executor: Đối tượng TradeExecutor
        
    Returns:
        Dict: Kết quả thực hiện
    """
    logger = get_logger('trade_command')
    
    # Tạo tín hiệu giao dịch
    trade_signal = {
        "symbol": args.symbol,
        "action": "buy" if args.side == "long" else "sell",
        "side": args.side,
        "position_size": args.size,
        "price": args.price,
        "order_type": args.type,
        "stop_loss": args.sl,
        "take_profit": args.tp,
        "leverage": args.leverage,
        "trailing_stop": args.trailing_stop
    }
    
    # Thực thi giao dịch
    logger.info(f"Mở vị thế {args.side} cho {args.symbol}: {args.size} @ {args.price if args.price else 'thị trường'}")
    result = trade_executor.execute_trade(trade_signal)
    
    return result

def close_position(args: argparse.Namespace, trade_executor: TradeExecutor) -> Dict[str, Any]:
    """
    Đóng vị thế.
    
    Args:
        args: Các tham số dòng lệnh
        trade_executor: Đối tượng TradeExecutor
        
    Returns:
        Dict: Kết quả thực hiện
    """
    logger = get_logger('trade_command')
    
    # Tạo tín hiệu giao dịch
    trade_signal = {
        "symbol": args.symbol,
        "action": "close",
        "position_size": args.size,
        "price": args.price
    }
    
    # Thực thi giao dịch
    logger.info(f"Đóng vị thế cho {args.symbol}{' với kích thước ' + str(args.size) if args.size else ''}")
    result = trade_executor.execute_trade(trade_signal)
    
    return result

def close_all_positions(args: argparse.Namespace, trade_executor: TradeExecutor) -> Dict[str, Any]:
    """
    Đóng tất cả vị thế.
    
    Args:
        args: Các tham số dòng lệnh
        trade_executor: Đối tượng TradeExecutor
        
    Returns:
        Dict: Kết quả thực hiện
    """
    logger = get_logger('trade_command')
    
    # Đóng tất cả vị thế
    logger.info(f"Đóng tất cả vị thế (Lý do: {args.reason})")
    result = trade_executor.close_all_positions(reason=args.reason)
    
    return result

def check_status(args: argparse.Namespace, trade_executor: TradeExecutor, 
                order_manager: OrderManager, account_manager: AccountManager) -> Dict[str, Any]:
    """
    Kiểm tra trạng thái vị thế/lệnh.
    
    Args:
        args: Các tham số dòng lệnh
        trade_executor: Đối tượng TradeExecutor
        order_manager: Đối tượng OrderManager
        account_manager: Đối tượng AccountManager
        
    Returns:
        Dict: Kết quả kiểm tra
    """
    logger = get_logger('trade_command')
    
    if args.order_id and args.symbol:
        # Kiểm tra trạng thái lệnh cụ thể
        logger.info(f"Kiểm tra trạng thái lệnh {args.order_id} cho {args.symbol}")
        return order_manager.fetch_order(args.order_id, args.symbol)
    elif args.symbol:
        # Kiểm tra vị thế cho một symbol
        logger.info(f"Kiểm tra vị thế cho {args.symbol}")
        position = trade_executor.get_position(args.symbol)
        return {"status": "success", "position": position}
    else:
        # Kiểm tra tất cả vị thế và thông tin tài khoản
        logger.info("Kiểm tra tất cả vị thế và thông tin tài khoản")
        
        result = {
            "positions": trade_executor.get_active_positions(),
            "account_balance": account_manager.get_balance(force_update=True),
            "performance": trade_executor.get_trading_performance()
        }
        
        return result

def cancel_order(args: argparse.Namespace, order_manager: OrderManager) -> Dict[str, Any]:
    """
    Hủy lệnh.
    
    Args:
        args: Các tham số dòng lệnh
        order_manager: Đối tượng OrderManager
        
    Returns:
        Dict: Kết quả thực hiện
    """
    logger = get_logger('trade_command')
    
    # Hủy lệnh
    logger.info(f"Hủy lệnh {args.order_id} cho {args.symbol}")
    result = order_manager.cancel_order(args.order_id, args.symbol)
    
    return result

def generate_report(args: argparse.Namespace, trade_executor: TradeExecutor) -> Dict[str, Any]:
    """
    Tạo báo cáo hiệu suất.
    
    Args:
        args: Các tham số dòng lệnh
        trade_executor: Đối tượng TradeExecutor
        
    Returns:
        Dict: Kết quả báo cáo
    """
    logger = get_logger('trade_command')
    
    # Tạo báo cáo
    logger.info(f"Tạo báo cáo hiệu suất cho {args.days} ngày qua")
    result = trade_executor.get_trading_performance(days=args.days)
    
    # Xuất báo cáo ra file nếu được chỉ định
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Đã xuất báo cáo vào file {args.output}")
    
    return result