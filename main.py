"""
Parser dòng lệnh cho hệ thống giao dịch tự động.
File này là điểm vào chính cho giao diện dòng lệnh, định nghĩa và xử lý các 
lệnh như dashboard, trade, collect, process, train, backtest, v.v.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Thêm thư mục gốc vào path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các module cần thiết
from config.logging_config import get_logger
from config.system_config import get_system_config

# Import các module lệnh
from cli.commands.dashboard_commands import setup_dashboard_parser
from cli.commands.trade_commands import setup_trade_parser
from cli.commands.collect_commands import setup_collect_parser
from cli.commands.process_commands import setup_process_parser
from cli.commands.train_commands import setup_train_parser

# Thiết lập logger
logger = get_logger('cli_parser')

def create_parser() -> argparse.ArgumentParser:
    """
    Tạo và cấu hình parser dòng lệnh.
    
    Returns:
        argparse.ArgumentParser: Parser đã được cấu hình
    """
    # Tạo parser chính
    parser = argparse.ArgumentParser(
        description='Hệ thống giao dịch tự động',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Thêm các tham số toàn cục
    parser.add_argument(
        '--config-file', 
        type=str,
        help='Đường dẫn đến file cấu hình'
    )
    
    parser.add_argument(
        '--log-level', 
        type=str, 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
        default='INFO',
        help='Mức độ log'
    )
    
    parser.add_argument(
        '--data-dir', 
        type=str,
        help='Thư mục chứa dữ liệu'
    )
    
    parser.add_argument(
        '--model-dir', 
        type=str,
        help='Thư mục chứa các mô hình'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str,
        help='Thư mục đầu ra'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='Automated Trading System v0.1.0',
        help='Hiển thị phiên bản và thoát'
    )
    
    # Tạo subparsers cho các lệnh chính
    subparsers = parser.add_subparsers(
        title='commands',
        description='Các lệnh có sẵn',
        dest='command'
    )
    
    # Thiết lập parsers cho từng lệnh
    setup_dashboard_parser(subparsers)
    setup_trade_parser(subparsers)
    setup_collect_parser(subparsers)
    setup_process_parser(subparsers)
    setup_train_parser(subparsers)
    
    # Backtest là trường hợp đặc biệt, cần xử lý riêng
    setup_backtest_parser(subparsers)
    
    return parser

def setup_backtest_parser(subparsers) -> None:
    """
    Thiết lập parser cho lệnh 'backtest'.
    
    Args:
        subparsers: Đối tượng subparsers từ argparse
    """
    backtest_parser = subparsers.add_parser(
        'backtest',
        help='Backtest chiến lược giao dịch',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Thêm các tham số cho lệnh backtest
    backtest_parser.add_argument(
        '--config', 
        type=str,
        help='Đường dẫn file cấu hình backtest'
    )
    
    backtest_parser.add_argument(
        '--data-dir', 
        type=str,
        help='Thư mục dữ liệu đầu vào'
    )
    
    backtest_parser.add_argument(
        '--output-dir', 
        type=str,
        help='Thư mục lưu kết quả backtest'
    )
    
    backtest_parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Hiển thị thông tin chi tiết'
    )
    
    # Thêm một số tham số cơ bản thường dùng trong backtest
    backtest_parser.add_argument(
        '--strategy', 
        type=str,
        help='Tên chiến lược để backtest'
    )
    
    backtest_parser.add_argument(
        '--symbols', 
        type=str,
        help='Danh sách cặp tiền (phân cách bằng dấu phẩy)'
    )
    
    backtest_parser.add_argument(
        '--start-date', 
        type=str,
        help='Ngày bắt đầu (định dạng YYYY-MM-DD)'
    )
    
    backtest_parser.add_argument(
        '--end-date', 
        type=str,
        help='Ngày kết thúc (định dạng YYYY-MM-DD)'
    )
    
    backtest_parser.add_argument(
        '--timeframe', 
        type=str, 
        default='1h',
        help='Khung thời gian (1m, 5m, 15m, 1h, 4h, 1d, ...)'
    )
    
    # Thiết lập hàm xử lý
    backtest_parser.set_defaults(func=handle_backtest_command)

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Phân tích tham số dòng lệnh.
    
    Args:
        args: Danh sách tham số dòng lệnh (mặc định là sys.argv[1:])
        
    Returns:
        argparse.Namespace: Các tham số đã phân tích
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Nếu không có lệnh nào được chỉ định, hiển thị help
    if not hasattr(parsed_args, 'command') or parsed_args.command is None:
        parser.print_help()
        sys.exit(1)
    
    return parsed_args

def setup_logging(log_level: str) -> None:
    """
    Thiết lập mức độ log cho hệ thống.
    
    Args:
        log_level: Mức độ log cần thiết lập
    """
    try:
        from config.logging_config import logging_config
        logging_config.set_global_level(log_level)
        logger.info(f"Đã cập nhật mức độ log: {log_level}")
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập mức độ log: {str(e)}")

def load_config(config_file: str) -> None:
    """
    Tải cấu hình từ file.
    
    Args:
        config_file: Đường dẫn đến file cấu hình
    """
    try:
        system_config = get_system_config()
        system_config.load_from_file(config_file)
        logger.info(f"Đã tải cấu hình từ {config_file}")
    except Exception as e:
        logger.error(f"Lỗi khi tải file cấu hình: {str(e)}")

def init_system(args: argparse.Namespace):
    """
    Khởi tạo hệ thống giao dịch tự động với các tham số đã cung cấp.
    
    Args:
        args: Các tham số dòng lệnh
        
    Returns:
        AutomatedTradingSystem: Hệ thống đã được khởi tạo
    """
    # Thiết lập logging
    if hasattr(args, 'log_level') and args.log_level:
        setup_logging(args.log_level)
    
    # Tải cấu hình
    if hasattr(args, 'config_file') and args.config_file:
        load_config(args.config_file)
    
    # Tạo instance của AutomatedTradingSystem
    try:
        from trading_system import AutomatedTradingSystem
        system = AutomatedTradingSystem()
        
        # Cập nhật thư mục dữ liệu và mô hình nếu được chỉ định
        if hasattr(args, 'data_dir') and args.data_dir:
            system.data_dir = Path(args.data_dir)
            logger.info(f"Sử dụng thư mục dữ liệu: {system.data_dir}")
        
        if hasattr(args, 'model_dir') and args.model_dir:
            system.model_dir = Path(args.model_dir)
            logger.info(f"Sử dụng thư mục mô hình: {system.model_dir}")
            
        if hasattr(args, 'output_dir') and args.output_dir:
            system.output_dir = Path(args.output_dir)
            logger.info(f"Sử dụng thư mục đầu ra: {system.output_dir}")
        
        return system
    
    except ImportError:
        logger.error("Không thể import AutomatedTradingSystem. Đảm bảo bạn đã cài đặt đúng dependencies.")
        sys.exit(1)

def handle_backtest_command(args: argparse.Namespace, system) -> int:
    """
    Xử lý lệnh 'backtest'.
    
    Args:
        args: Các tham số dòng lệnh
        system: Instance của AutomatedTradingSystem
        
    Returns:
        int: Mã kết quả (0 = thành công)
    """
    try:
        # Sử dụng module backtest_commands
        from cli.commands.backtest_commands import BacktestCommands
        
        # Khởi tạo BacktestCommands
        cmd = BacktestCommands(
            data_dir=args.data_dir if hasattr(args, 'data_dir') and args.data_dir else None,
            output_dir=args.output_dir if hasattr(args, 'output_dir') and args.output_dir else None,
            config_path=args.config if hasattr(args, 'config') and args.config else None,
            verbose=args.verbose if hasattr(args, 'verbose') else False
        )
        
        # Thực hiện backtest
        if hasattr(args, 'strategy') and args.strategy:
            # Tìm hàm chiến lược
            strategy_paths = [
                Path("strategies"),
                Path("trading_strategies"),
                Path("custom_strategies")
            ]
            strategy_name = args.strategy
            strategy_found = False
            
            for path in strategy_paths:
                strategy_file = path / f"{strategy_name}.py"
                if strategy_file.exists():
                    # Import chiến lược
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(strategy_name, strategy_file)
                    strategy_module = importlib.util.module_from_spec(spec)
                    sys.modules[strategy_name] = strategy_module
                    spec.loader.exec_module(strategy_module)
                    
                    # Tìm hàm chiến lược
                    strategy_func = None
                    for attr_name in dir(strategy_module):
                        if attr_name.startswith("strategy_") or attr_name == strategy_name:
                            strategy_func = getattr(strategy_module, attr_name)
                            if callable(strategy_func):
                                strategy_found = True
                                break
                    
                    if strategy_found:
                        # Đăng ký chiến lược
                        cmd.register_strategy(
                            strategy_func=strategy_func,
                            strategy_name=strategy_name
                        )
                        
                        # Chuyển đổi symbols nếu cần
                        symbols = args.symbols.split(',') if hasattr(args, 'symbols') and args.symbols else ["BTC/USDT"]
                        timeframe = args.timeframe if hasattr(args, 'timeframe') else "1h"
                        
                        # Tải dữ liệu
                        data = cmd.load_data(
                            symbols=symbols,
                            timeframe=timeframe,
                            start_date=args.start_date if hasattr(args, 'start_date') else None,
                            end_date=args.end_date if hasattr(args, 'end_date') else None
                        )
                        
                        # Chạy backtest
                        result = cmd.run_backtest(
                            strategy_name=strategy_name,
                            data=data
                        )
                        
                        # Đánh giá chiến lược
                        cmd.evaluate_strategy(result, strategy_name)
                        
                        return 0
                        
            if not strategy_found:
                logger.error(f"Không tìm thấy chiến lược {strategy_name} trong các thư mục chiến lược")
        
        # Trong trường hợp không có chiến lược cụ thể hoặc không tìm thấy chiến lược
        logger.info("Khởi động giao diện backtest chung")
        # Ở đây có thể khởi động giao diện tìm chiến lược có sẵn
        return 0
        
    except ImportError as e:
        logger.error(f"Không thể import BacktestCommands: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện backtest: {str(e)}", exc_info=True)
        return 1

def main() -> int:
    """
    Hàm chính để xử lý dòng lệnh.
    
    Returns:
        int: Mã kết quả (0 = thành công)
    """
    try:
        # Phân tích tham số dòng lệnh
        args = parse_args()
        
        # Khởi tạo hệ thống
        system = init_system(args)
        
        # Gọi hàm xử lý tương ứng
        if hasattr(args, 'func'):
            try:
                result = args.func(args, system)
                # Trả về mã kết quả nếu hàm trả về int, mặc định là 0
                return result if isinstance(result, int) else 0
            except Exception as e:
                logger.error(f"Lỗi khi thực hiện lệnh {args.command}: {str(e)}", exc_info=True)
                return 1
        else:
            logger.error(f"Không tìm thấy handler cho lệnh {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Đã hủy thao tác bởi người dùng")
        return 130
    except Exception as e:
        logger.error(f"Lỗi không xác định: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())