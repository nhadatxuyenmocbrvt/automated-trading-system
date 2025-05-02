"""
Xử lý lệnh dashboard.
File này cung cấp hàm xử lý lệnh 'dashboard' từ CLI.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
import logging
import subprocess
import signal

from config.logging_config import get_logger
from trading_system import AutomatedTradingSystem

# Thiết lập logger
logger = get_logger("dashboard_commands")

async def handle_dashboard_command(args, trading_system):
    """
    Xử lý lệnh dashboard từ CLI.
    
    Args:
        args: Đối tượng ArgumentParser đã parse
        trading_system: Instance của AutomatedTradingSystem
    """
    logger.info("Đang khởi chạy dashboard")
    
    # Kiểm tra xem Streamlit đã được cài đặt chưa
    try:
        import streamlit
        streamlit_installed = True
    except ImportError:
        streamlit_installed = False
        logger.error("Streamlit chưa được cài đặt. Vui lòng cài đặt với lệnh: pip install streamlit")
        logger.info("Hoặc bạn có thể cài đặt đầy đủ các gói phụ thuộc: pip install -r requirements_dashboard.txt")
        return
    
    # Xác định đường dẫn đến file app Streamlit
    dashboard_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "streamlit_dashboard",
        "app.py"
    )
    
    if not os.path.exists(dashboard_path):
        logger.error(f"Không tìm thấy file dashboard: {dashboard_path}")
        return
    
    # Xây dựng các tham số cho Streamlit
    streamlit_args = ["streamlit", "run", dashboard_path]
    
    # Thêm cổng
    streamlit_args.extend(["--server.port", str(args.port)])
    
    # Thêm các tham số khác
    env_vars = {}
    
    if args.data_dir:
        env_vars["TRADING_DATA_DIR"] = os.path.abspath(args.data_dir)
    
    if args.model_dir:
        env_vars["TRADING_MODEL_DIR"] = os.path.abspath(args.model_dir)
    
    if args.log_dir:
        env_vars["TRADING_LOG_DIR"] = os.path.abspath(args.log_dir)
    
    if args.config_file:
        env_vars["TRADING_CONFIG_FILE"] = os.path.abspath(args.config_file)
    
    logger.info(f"Khởi chạy dashboard Streamlit trên cổng {args.port}")
    
    # Cập nhật environment variables
    env = os.environ.copy()
    env.update(env_vars)
    
    # Chạy Streamlit như một subprocess
    try:
        process = subprocess.Popen(
            streamlit_args,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        logger.info(f"Dashboard đang chạy với PID {process.pid}")
        logger.info("Truy cập dashboard tại: http://localhost:%s", args.port)
        logger.info("Nhấn Ctrl+C để dừng dashboard")
        
        # Xử lý output từ Streamlit
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
            
            error = process.stderr.readline()
            if error and error.strip():
                logger.error(error.strip())
        
        # Đợi process kết thúc
        return_code = process.poll()
        if return_code != 0:
            logger.error(f"Dashboard bị lỗi với mã trả về {return_code}")
        
    except KeyboardInterrupt:
        logger.info("Đã nhận lệnh thoát từ người dùng (Ctrl+C)")
        logger.info("Đang dừng dashboard...")
        
        # Gửi signal để kết thúc process
        if process.poll() is None:  # Process vẫn đang chạy
            if sys.platform == "win32":
                # Windows
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # Linux/Mac
                process.send_signal(signal.SIGTERM)
            
            # Đợi process kết thúc
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Nếu quá thời gian, kill process
                logger.warning("Process không kết thúc, buộc dừng...")
                process.kill()
        
        logger.info("Dashboard đã dừng")
    
    except Exception as e:
        logger.error(f"Lỗi khi chạy dashboard: {str(e)}")
        
        # Đảm bảo process được dừng khi có lỗi
        if 'process' in locals() and process.poll() is None:
            process.kill()
        
        raise