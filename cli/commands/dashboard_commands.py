"""
Xử lý lệnh dashboard.
File này định nghĩa các tham số và xử lý cho lệnh 'dashboard' trên CLI.
"""

import os
import sys
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

# Thêm thư mục gốc vào path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import các module cần thiết
from config.logging_config import get_logger
from trading_system import AutomatedTradingSystem

def setup_dashboard_parser(subparsers) -> None:
    """
    Thiết lập parser cho lệnh 'dashboard'.
    
    Args:
        subparsers: Subparsers object từ argparse
    """
    dashboard_parser = subparsers.add_parser(
        'dashboard',
        help='Khởi động dashboard theo dõi',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Cấu hình port
    dashboard_parser.add_argument(
        '--port', 
        type=int, 
        default=8501,
        help='Cổng chạy dashboard Streamlit'
    )
    
    # Chế độ hiển thị
    dashboard_parser.add_argument(
        '--mode', 
        type=str, 
        choices=['light', 'dark'], 
        default='light',
        help='Chế độ giao diện light/dark'
    )
    
    dashboard_parser.add_argument(
        '--data-dir', 
        type=str,
        help='Thư mục chứa dữ liệu'
    )
    
    dashboard_parser.add_argument(
        '--model-dir', 
        type=str,
        help='Thư mục chứa các mô hình'
    )
    
    dashboard_parser.set_defaults(func=handle_dashboard_command)

def handle_dashboard_command(args: argparse.Namespace, system: AutomatedTradingSystem) -> int:
    """
    Xử lý lệnh 'dashboard'.
    
    Args:
        args: Các tham số dòng lệnh
        system: Instance của AutomatedTradingSystem
        
    Returns:
        int: Mã kết quả (0 = thành công)
    """
    logger = get_logger('dashboard_command')
    
    try:
        # Xử lý tham số đầu vào
        port = args.port
        mode = args.mode
        data_dir = args.data_dir
        model_dir = args.model_dir
        
        # Cập nhật thư mục dữ liệu nếu cần
        if data_dir:
            system.data_dir = Path(data_dir)
        
        if model_dir:
            system.model_dir = Path(model_dir)
        
        # Khởi động dashboard
        logger.info(f"Khởi động dashboard tại http://localhost:{port}")
        logger.info(f"Chế độ giao diện: {mode}")
        
        # Thiết lập biến môi trường cho Streamlit
        os.environ["STREAMLIT_THEME"] = mode
        
        # Khởi động dashboard
        system.start_dashboard(port=port)
        
        # Đợi người dùng hủy
        logger.info("Dashboard đang chạy. Nhấn Ctrl+C để dừng.")
        
        # Giữ chương trình chạy
        try:
            from threading import Event
            Event().wait()
        except KeyboardInterrupt:
            logger.info("Đã nhận tín hiệu ngắt. Đóng dashboard...")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Đóng dashboard")
        return 130
    except Exception as e:
        logger.error(f"Lỗi khi khởi động dashboard: {str(e)}", exc_info=True)
        return 1