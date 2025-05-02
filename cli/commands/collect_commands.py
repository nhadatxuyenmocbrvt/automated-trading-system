"""
Xử lý lệnh thu thập dữ liệu.
File này định nghĩa các tham số và xử lý cho lệnh 'collect' trên CLI.
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

def setup_collect_parser(subparsers) -> None:
    """
    Thiết lập parser cho lệnh 'collect'.
    
    Args:
        subparsers: Subparsers object từ argparse
    """
    collect_parser = subparsers.add_parser(
        'collect',
        help='Thu thập dữ liệu thị trường',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Tham số bắt buộc
    collect_parser.add_argument(
        '--exchange', 
        type=str, 
        required=True,
        help='Sàn giao dịch (binance, bybit, ...)'
    )
    
    collect_parser.add_argument(
        '--symbols', 
        type=str, 
        required=True, 
        nargs='+',
        help='Danh sách cặp giao dịch (BTC/USDT ETH/USDT ...)'
    )
    
    # Tham số thời gian và khung thời gian
    collect_parser.add_argument(
        '--timeframes', 
        type=str, 
        nargs='+', 
        default=['1h'],
        help='Khung thời gian (1m, 5m, 15m, 1h, 4h, 1d, ...)'
    )
    
    time_group = collect_parser.add_mutually_exclusive_group()
    
    time_group.add_argument(
        '--days', 
        type=int, 
        help='Số ngày dữ liệu cần thu thập (tính từ hiện tại)'
    )
    
    time_group.add_argument(
        '--start-date', 
        type=str, 
        help='Ngày bắt đầu (YYYY-MM-DD)'
    )
    
    collect_parser.add_argument(
        '--end-date', 
        type=str, 
        help='Ngày kết thúc (YYYY-MM-DD), mặc định là hiện tại'
    )
    
    # Loại thị trường
    collect_parser.add_argument(
        '--futures', 
        action='store_true',
        help='Thu thập dữ liệu futures thay vì spot'
    )
    
    # Thư mục đầu ra
    collect_parser.add_argument(
        '--output-dir', 
        type=str,
        help='Thư mục lưu dữ liệu'
    )
    
    collect_parser.set_defaults(func=handle_collect_command)

def handle_collect_command(args: argparse.Namespace, system: AutomatedTradingSystem) -> int:
    """
    Xử lý lệnh 'collect'.
    
    Args:
        args: Các tham số dòng lệnh
        system: Instance của AutomatedTradingSystem
        
    Returns:
        int: Mã kết quả (0 = thành công)
    """
    logger = get_logger('collect_command')
    
    try:
        # Xử lý tham số đầu vào
        exchange_id = args.exchange.lower()
        symbols = args.symbols
        timeframes = args.timeframes
        days = args.days
        start_date = args.start_date
        end_date = args.end_date
        futures = args.futures
        output_dir = args.output_dir
        
        # Chuyển đổi thành đường dẫn nếu có
        if output_dir:
            output_dir = Path(output_dir)
        
        # Xử lý thu thập dữ liệu cho mỗi khung thời gian
        result_paths = {}
        
        for timeframe in timeframes:
            logger.info(f"Thu thập dữ liệu cho khung thời gian {timeframe}")
            
            # Chạy thu thập dữ liệu bất đồng bộ
            paths = asyncio.run(system.collect_data(
                exchange_id=exchange_id,
                symbols=symbols,
                timeframe=timeframe,
                days=days,
                start_date=start_date,
                end_date=end_date,
                futures=futures,
                output_dir=output_dir
            ))
            
            if paths:
                result_paths[timeframe] = paths
                symbol_count = len(paths)
                logger.info(f"Đã thu thập dữ liệu cho {symbol_count} cặp tiền với khung thời gian {timeframe}")
                
                # In ra đường dẫn tới dữ liệu
                for symbol, path in paths.items():
                    logger.info(f"  - {symbol}: {path}")
            else:
                logger.warning(f"Không có dữ liệu nào được thu thập cho khung thời gian {timeframe}")
        
        # Tóm tắt kết quả
        if result_paths:
            total_timeframes = len(result_paths)
            total_symbols = sum(len(paths) for paths in result_paths.values())
            logger.info(f"Tổng kết: Đã thu thập {total_symbols} datasets cho {total_timeframes} khung thời gian")
            return 0
        else:
            logger.error("Không thu thập được dữ liệu nào")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Đã hủy thu thập dữ liệu")
        return 130
    except Exception as e:
        logger.error(f"Lỗi khi thu thập dữ liệu: {str(e)}", exc_info=True)
        return 1