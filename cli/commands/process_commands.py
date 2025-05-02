"""
Xử lý lệnh xử lý dữ liệu.
File này định nghĩa các tham số và xử lý cho lệnh 'process' trên CLI.
"""

import os
import sys
import argparse
import asyncio
import logging  # Thêm dòng này
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
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

def _prepare_directories(input_dir: Optional[str], 
                         output_dir: Optional[str], 
                         system: AutomatedTradingSystem, 
                         default_input_subdir: str = "collected") -> Tuple[Path, Optional[Path]]:
    """
    Chuẩn bị và kiểm tra các thư mục đầu vào/đầu ra.
    
    Args:
        input_dir: Thư mục đầu vào
        output_dir: Thư mục đầu ra
        system: Instance của AutomatedTradingSystem
        default_input_subdir: Thư mục con mặc định
        
    Returns:
        Tuple (input_dir_path, output_dir_path)
    """
    # Chuyển đổi thành đường dẫn nếu có
    if input_dir:
        input_dir_path = Path(input_dir)
    else:
        input_dir_path = system.data_dir / default_input_subdir
    
    # Chuyển đổi output_dir nếu có
    output_dir_path = None
    if output_dir:
        output_dir_path = Path(output_dir)
    
    return input_dir_path, output_dir_path

def _find_data_files(input_dir: Path, 
                    symbols: Optional[List[str]], 
                    timeframes: List[str], 
                    logger: logging.Logger) -> Dict[str, Path]:
    """
    Tìm các file dữ liệu phù hợp với symbols và timeframes.
    
    Args:
        input_dir: Thư mục đầu vào
        symbols: Danh sách cặp giao dịch
        timeframes: Danh sách khung thời gian
        logger: Logger
        
    Returns:
        Dict với key là symbol và value là đường dẫn file
    """
    data_paths = {}
    
    # Duyệt qua các thư mục con để tìm file
    if input_dir.exists():
        for timeframe in timeframes:
            # Tìm các file phù hợp với timeframe
            if symbols:
                for symbol in symbols:
                    symbol_safe = symbol.replace('/', '_').lower()
                    
                    # Thử các pattern khác nhau, từ cụ thể đến linh hoạt
                    patterns = [
                        f"**/{symbol_safe}_{timeframe}*.parquet",  # Pattern cụ thể
                        f"**/{symbol_safe}_*.parquet",             # Pattern chỉ theo symbol
                        f"**/*{symbol_safe}*{timeframe}*.parquet", # Pattern hỗn hợp
                        f"**/*{symbol_safe}*.parquet"              # Pattern linh hoạt
                    ]
                    
                    for pattern in patterns:
                        logger.debug(f"Tìm kiếm với pattern: {pattern}")
                        matching_files = list(input_dir.glob(pattern))
                        
                        if matching_files:
                            logger.info(f"Tìm thấy {len(matching_files)} file với pattern {pattern}")
                            # Sắp xếp theo thời gian tạo để lấy file mới nhất
                            newest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
                            data_paths[symbol] = newest_file
                            logger.info(f"Sử dụng file mới nhất cho {symbol}: {newest_file}")
                            break
            else:
                # Nếu không có symbols, tìm tất cả các file
                pattern = f"**/*{timeframe}*.parquet"
                matching_files = list(input_dir.glob(pattern))
                
                for file_path in matching_files:
                    # Trích xuất symbol từ tên file
                    filename = file_path.stem
                    parts = filename.split('_')
                    # Tạo symbol từ hai phần đầu tiên nếu có (btc_usdt -> BTC/USDT)
                    if len(parts) >= 2:
                        symbol = f"{parts[0].upper()}/{parts[1].upper()}"
                        data_paths[symbol] = file_path
                        logger.info(f"Tìm thấy dữ liệu cho {symbol}: {file_path}")
    
    return data_paths

def _process_and_report_results(system: AutomatedTradingSystem, 
                               data_paths: Dict[str, Path], 
                               clean_data: bool, 
                               generate_features: bool, 
                               pipeline_name: Optional[str] = None,
                               output_dir: Optional[Path] = None, 
                               logger: logging.Logger = None) -> int:
    """
    Xử lý dữ liệu và báo cáo kết quả.
    
    Args:
        system: Instance của AutomatedTradingSystem
        data_paths: Dict với key là symbol và value là đường dẫn file
        clean_data: Có làm sạch dữ liệu không
        generate_features: Có tạo đặc trưng không
        pipeline_name: Tên pipeline xử lý
        output_dir: Thư mục đầu ra
        logger: Logger
        
    Returns:
        Mã kết quả (0 = thành công)
    """
    if not data_paths:
        logger.error(f"Không tìm thấy file dữ liệu phù hợp")
        return 1
    
    # Xử lý dữ liệu
    result_paths = system.process_data(
        data_paths=data_paths,
        pipeline_name=pipeline_name,
        clean_data=clean_data,
        generate_features=generate_features,
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
        # Chuẩn bị thư mục đầu vào/ra
        input_dir, output_dir = _prepare_directories(args.input_dir, args.output_dir, system)
        
        # Tìm các file dữ liệu
        data_paths = _find_data_files(input_dir, args.symbols, args.timeframes, logger)
        
        # Xử lý và báo cáo kết quả
        return _process_and_report_results(
            system, data_paths, 
            clean_data=True, 
            generate_features=False,
            output_dir=output_dir,
            logger=logger
        )
            
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
        # Chuẩn bị thư mục đầu vào/ra
        input_dir, output_dir = _prepare_directories(
            args.input_dir, 
            args.output_dir, 
            system, 
            default_input_subdir="processed"
        )
        
        # Nếu thư mục processed không tồn tại, thử sử dụng thư mục collected
        if not input_dir.exists():
            input_dir, _ = _prepare_directories(None, None, system, default_input_subdir="collected")
            logger.info(f"Thư mục processed không tồn tại, sử dụng thư mục {input_dir}")
        
        # Tìm các file dữ liệu
        data_paths = _find_data_files(input_dir, args.symbols, [''], logger)  # Không filter theo timeframe
        
        # Xử lý và báo cáo kết quả
        return _process_and_report_results(
            system, data_paths, 
            clean_data=False, 
            generate_features=True,
            output_dir=output_dir,
            logger=logger
        )
            
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
        # Chuẩn bị thư mục đầu vào/ra
        input_dir, output_dir = _prepare_directories(args.input_dir, args.output_dir, system)
        
        # Tìm các file dữ liệu
        data_paths = _find_data_files(input_dir, args.symbols, args.timeframes, logger)
        
        # Xử lý và báo cáo kết quả
        return _process_and_report_results(
            system, data_paths, 
            clean_data=not args.no_clean, 
            generate_features=not args.no_features,
            pipeline_name=args.pipeline_name,
            output_dir=output_dir,
            logger=logger
        )
            
    except Exception as e:
        logger.error(f"Lỗi khi chạy pipeline xử lý dữ liệu: {str(e)}", exc_info=True)
        return 1