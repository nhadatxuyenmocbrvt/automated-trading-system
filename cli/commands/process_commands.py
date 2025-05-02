"""
Xử lý lệnh xử lý dữ liệu.
File này định nghĩa các tham số và xử lý cho lệnh 'process' trên CLI.
"""

import os
import sys
import argparse
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Thêm thư mục gốc vào path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import các module cần thiết
from config.logging_config import get_logger
from trading_system import AutomatedTradingSystem

def setup_process_parser(subparsers) -> None:
    """
    Thiết lập parser cho lệnh 'process'.
    
    Args:
        subparsers: Subparsers object từ argparse
    """
    process_parser = subparsers.add_parser(
        'process',
        help='Xử lý dữ liệu thị trường',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Tạo subparsers cho các lệnh con
    process_subparsers = process_parser.add_subparsers(title='subcommands', dest='process_command')
    
    # 1. Lệnh làm sạch dữ liệu
    clean_parser = process_subparsers.add_parser(
        'clean',
        help='Làm sạch dữ liệu thị trường',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    clean_parser.add_argument(
        '--data-type', 
        type=str, 
        choices=['ohlcv', 'trades', 'orderbook', 'all'], 
        default='ohlcv',
        help='Loại dữ liệu cần làm sạch'
    )
    
    clean_parser.add_argument(
        '--input-dir', 
        type=str,
        help='Thư mục chứa dữ liệu đầu vào'
    )
    
    clean_parser.add_argument(
        '--symbols', 
        type=str, 
        nargs='+',
        help='Danh sách cặp giao dịch cần xử lý'
    )
    
    clean_parser.add_argument(
        '--timeframes', 
        type=str, 
        nargs='+', 
        default=['1h'],
        help='Khung thời gian cần xử lý'
    )
    
    clean_parser.add_argument(
        '--output-dir', 
        type=str,
        help='Thư mục lưu dữ liệu đã làm sạch'
    )
    
    # 2. Lệnh tạo đặc trưng
    features_parser = process_subparsers.add_parser(
        'features',
        help='Tạo đặc trưng từ dữ liệu thị trường',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    features_parser.add_argument(
        '--data-type', 
        type=str, 
        choices=['ohlcv', 'all'], 
        default='ohlcv',
        help='Loại dữ liệu cần tạo đặc trưng'
    )
    
    features_parser.add_argument(
        '--input-dir', 
        type=str,
        help='Thư mục chứa dữ liệu đầu vào'
    )
    
    features_parser.add_argument(
        '--symbols', 
        type=str, 
        nargs='+',
        help='Danh sách cặp giao dịch cần xử lý'
    )
    
    features_parser.add_argument(
        '--indicators', 
        type=str, 
        nargs='+',
        help='Danh sách chỉ báo kỹ thuật cần tạo'
    )
    
    features_parser.add_argument(
        '--all-indicators', 
        action='store_true',
        help='Tạo tất cả các chỉ báo kỹ thuật có sẵn'
    )
    
    features_parser.add_argument(
        '--output-dir', 
        type=str,
        help='Thư mục lưu dữ liệu đã tạo đặc trưng'
    )
    
    # 3. Lệnh pipeline (làm sạch và tạo đặc trưng)
    pipeline_parser = process_subparsers.add_parser(
        'pipeline',
        help='Chạy toàn bộ pipeline xử lý dữ liệu',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    pipeline_parser.add_argument(
        '--input-dir', 
        type=str,
        help='Thư mục chứa dữ liệu đầu vào'
    )
    
    pipeline_parser.add_argument(
        '--symbols', 
        type=str, 
        nargs='+',
        help='Danh sách cặp giao dịch cần xử lý'
    )
    
    pipeline_parser.add_argument(
        '--timeframes', 
        type=str, 
        nargs='+', 
        default=['1h'],
        help='Khung thời gian cần xử lý'
    )
    
    pipeline_parser.add_argument(
        '--start-date', 
        type=str, 
        help='Ngày bắt đầu (YYYY-MM-DD)'
    )
    
    pipeline_parser.add_argument(
        '--end-date', 
        type=str, 
        help='Ngày kết thúc (YYYY-MM-DD)'
    )
    
    pipeline_parser.add_argument(
        '--output-dir', 
        type=str,
        help='Thư mục lưu dữ liệu đã xử lý'
    )
    
    pipeline_parser.add_argument(
        '--pipeline-name', 
        type=str,
        help='Tên của pipeline xử lý (nếu sử dụng pipeline đã đăng ký)'
    )
    
    pipeline_parser.add_argument(
        '--no-clean', 
        action='store_true',
        help='Bỏ qua bước làm sạch dữ liệu'
    )
    
    pipeline_parser.add_argument(
        '--no-features', 
        action='store_true',
        help='Bỏ qua bước tạo đặc trưng'
    )
    
    process_parser.set_defaults(func=handle_process_command)

def handle_process_command(args: argparse.Namespace, system: AutomatedTradingSystem) -> int:
    """
    Xử lý lệnh 'process'.
    
    Args:
        args: Các tham số dòng lệnh
        system: Instance của AutomatedTradingSystem
        
    Returns:
        int: Mã kết quả (0 = thành công)
    """
    logger = get_logger('process_command')
    
    try:
        # Nếu không có lệnh con, hiển thị help
        if not hasattr(args, 'process_command') or args.process_command is None:
            logger.error("Thiếu lệnh con. Sử dụng một trong các lệnh: clean, features, pipeline")
            return 1
        
        # Xử lý từng lệnh con
        process_command = args.process_command
        
        if process_command == 'clean':
            return handle_clean_command(args, system)
        elif process_command == 'features':
            return handle_features_command(args, system)
        elif process_command == 'pipeline':
            return handle_pipeline_command(args, system)
        else:
            logger.error(f"Lệnh con không hợp lệ: {process_command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Đã hủy xử lý dữ liệu")
        return 130
    except Exception as e:
        logger.error(f"Lỗi khi xử lý dữ liệu: {str(e)}", exc_info=True)
        return 1

def handle_clean_command(args: argparse.Namespace, system: AutomatedTradingSystem) -> int:
    """
    Xử lý lệnh 'process clean'.
    
    Args:
        args: Các tham số dòng lệnh
        system: Instance của AutomatedTradingSystem
        
    Returns:
        int: Mã kết quả (0 = thành công)
    """
    logger = get_logger('process_clean')
    
    try:
        # Xử lý tham số đầu vào
        data_type = args.data_type
        input_dir = args.input_dir
        symbols = args.symbols
        timeframes = args.timeframes
        output_dir = args.output_dir
        
        # Chuyển đổi thành đường dẫn nếu có
        if input_dir:
            input_dir = Path(input_dir)
        if output_dir:
            output_dir = Path(output_dir)
        
        # Nếu không có input_dir, sử dụng thư mục mặc định
        if not input_dir:
            input_dir = system.data_dir / "collected"
        
        # Tìm tất cả các file dữ liệu phù hợp
        data_paths = {}
        
        # Duyệt qua các thư mục con để tìm file
        if input_dir.exists():
            for timeframe in timeframes:
                # Tìm các file phù hợp với timeframe
                if symbols:
                    for symbol in symbols:
                        symbol_safe = symbol.replace('/', '_')
                        pattern = f"*{symbol_safe}*{timeframe}*.parquet"
                        matching_files = list(input_dir.glob(f"**/{pattern}"))
                        
                        if matching_files:
                            data_paths[symbol] = matching_files[0]
                else:
                    # Nếu không có symbols, tìm tất cả các file
                    pattern = f"*{timeframe}*.parquet"
                    matching_files = list(input_dir.glob(f"**/{pattern}"))
                    
                    for file_path in matching_files:
                        # Trích xuất symbol từ tên file
                        filename = file_path.stem
                        parts = filename.split('_')
                        # Giả sử symbol nằm ở phần đầu tiên
                        if parts:
                            symbol = parts[0]
                            data_paths[symbol] = file_path
        
        if not data_paths:
            logger.error(f"Không tìm thấy file dữ liệu nào trong {input_dir}")
            return 1
        
        # Xử lý dữ liệu
        result_paths = system.process_data(
            data_paths=data_paths,
            clean_data=True,
            generate_features=False,
            output_dir=output_dir
        )
        
        # Tóm tắt kết quả
        if result_paths:
            total_symbols = len(result_paths)
            logger.info(f"Tổng kết: Đã làm sạch dữ liệu cho {total_symbols} cặp tiền")
            
            # In ra đường dẫn tới dữ liệu
            for symbol, path in result_paths.items():
                logger.info(f"  - {symbol}: {path}")
                
            return 0
        else:
            logger.error("Không có dữ liệu nào được xử lý")
            return 1
            
    except Exception as e:
        logger.error(f"Lỗi khi làm sạch dữ liệu: {str(e)}", exc_info=True)
        return 1

def handle_features_command(args: argparse.Namespace, system: AutomatedTradingSystem) -> int:
    """
    Xử lý lệnh 'process features'.
    
    Args:
        args: Các tham số dòng lệnh
        system: Instance của AutomatedTradingSystem
        
    Returns:
        int: Mã kết quả (0 = thành công)
    """
    logger = get_logger('process_features')
    
    try:
        # Xử lý tham số đầu vào
        data_type = args.data_type
        input_dir = args.input_dir
        symbols = args.symbols
        indicators = args.indicators
        all_indicators = args.all_indicators
        output_dir = args.output_dir
        
        # Chuyển đổi thành đường dẫn nếu có
        if input_dir:
            input_dir = Path(input_dir)
        if output_dir:
            output_dir = Path(output_dir)
        
        # Nếu không có input_dir, sử dụng thư mục mặc định
        if not input_dir:
            input_dir = system.data_dir / "processed"
            
            # Nếu không tồn tại, thử sử dụng thư mục collected
            if not input_dir.exists():
                input_dir = system.data_dir / "collected"
        
        # Tìm tất cả các file dữ liệu phù hợp
        data_paths = {}
        
        # Duyệt qua các thư mục con để tìm file
        if input_dir.exists():
            if symbols:
                for symbol in symbols:
                    symbol_safe = symbol.replace('/', '_')
                    pattern = f"*{symbol_safe}*.parquet"
                    matching_files = list(input_dir.glob(f"**/{pattern}"))
                    
                    if matching_files:
                        data_paths[symbol] = matching_files[0]
            else:
                # Nếu không có symbols, tìm tất cả các file
                pattern = "*.parquet"
                matching_files = list(input_dir.glob(f"**/{pattern}"))
                
                for file_path in matching_files:
                    # Trích xuất symbol từ tên file
                    filename = file_path.stem
                    parts = filename.split('_')
                    # Giả sử symbol nằm ở phần đầu tiên
                    if parts:
                        symbol = parts[0]
                        data_paths[symbol] = file_path
        
        if not data_paths:
            logger.error(f"Không tìm thấy file dữ liệu nào trong {input_dir}")
            return 1
        
        # Xử lý dữ liệu
        result_paths = system.process_data(
            data_paths=data_paths,
            clean_data=False,
            generate_features=True,
            output_dir=output_dir
        )
        
        # Tóm tắt kết quả
        if result_paths:
            total_symbols = len(result_paths)
            logger.info(f"Tổng kết: Đã tạo đặc trưng cho {total_symbols} cặp tiền")
            
            # In ra đường dẫn tới dữ liệu
            for symbol, path in result_paths.items():
                logger.info(f"  - {symbol}: {path}")
                
            return 0
        else:
            logger.error("Không có dữ liệu nào được xử lý")
            return 1
            
    except Exception as e:
        logger.error(f"Lỗi khi tạo đặc trưng: {str(e)}", exc_info=True)
        return 1

def handle_pipeline_command(args: argparse.Namespace, system: AutomatedTradingSystem) -> int:
    """
    Xử lý lệnh 'process pipeline'.
    
    Args:
        args: Các tham số dòng lệnh
        system: Instance của AutomatedTradingSystem
        
    Returns:
        int: Mã kết quả (0 = thành công)
    """
    logger = get_logger('process_pipeline')
    
    try:
        # Xử lý tham số đầu vào
        input_dir = args.input_dir
        symbols = args.symbols
        timeframes = args.timeframes
        start_date = args.start_date
        end_date = args.end_date
        output_dir = args.output_dir
        pipeline_name = args.pipeline_name
        no_clean = args.no_clean
        no_features = args.no_features
        
        # Chuyển đổi thành đường dẫn nếu có
        if input_dir:
            input_dir = Path(input_dir)
        if output_dir:
            output_dir = Path(output_dir)
        
        # Nếu không có input_dir, sử dụng thư mục mặc định
        if not input_dir:
            input_dir = system.data_dir / "collected"
        
        # Tìm tất cả các file dữ liệu phù hợp
        data_paths = {}
        
        # Duyệt qua các thư mục con để tìm file
        if input_dir.exists():
            for timeframe in timeframes:
                # Tìm các file phù hợp với timeframe
                if symbols:
                    for symbol in symbols:
                        symbol_safe = symbol.replace('/', '_')
                        pattern = f"*{symbol_safe}*{timeframe}*.parquet"
                        matching_files = list(input_dir.glob(f"**/{pattern}"))
                        
                        if matching_files:
                            data_paths[symbol] = matching_files[0]
                else:
                    # Nếu không có symbols, tìm tất cả các file
                    pattern = f"*{timeframe}*.parquet"
                    matching_files = list(input_dir.glob(f"**/{pattern}"))
                    
                    for file_path in matching_files:
                        # Trích xuất symbol từ tên file
                        filename = file_path.stem
                        parts = filename.split('_')
                        # Giả sử symbol nằm ở phần đầu tiên
                        if parts:
                            symbol = parts[0]
                            data_paths[symbol] = file_path
        
        if not data_paths:
            logger.error(f"Không tìm thấy file dữ liệu nào trong {input_dir}")
            return 1
        
        # Xử lý dữ liệu
        result_paths = system.process_data(
            data_paths=data_paths,
            pipeline_name=pipeline_name,
            clean_data=not no_clean,
            generate_features=not no_features,
            output_dir=output_dir
        )
        
        # Tóm tắt kết quả
        if result_paths:
            total_symbols = len(result_paths)
            logger.info(f"Tổng kết: Đã xử lý dữ liệu cho {total_symbols} cặp tiền")
            
            # In ra đường dẫn tới dữ liệu
            for symbol, path in result_paths.items():
                logger.info(f"  - {symbol}: {path}")
                
            return 0
        else:
            logger.error("Không có dữ liệu nào được xử lý")
            return 1
            
    except Exception as e:
        logger.error(f"Lỗi khi chạy pipeline xử lý dữ liệu: {str(e)}", exc_info=True)
        return 1