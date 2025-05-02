"""
Lớp quản lý chính của hệ thống giao dịch tự động.
"""

import asyncio
import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from config.system_config import get_system_config
from config.logging_config import get_logger, setup_component_logger
from config.env import get_env, env_manager
from config.constants import SystemStatus, Exchange, Indicator

# Import các module thu thập dữ liệu
from data_collectors.exchange_api.binance_connector import BinanceConnector
from data_collectors.exchange_api.generic_connector import ExchangeConnector
from data_collectors.market_data.historical_data_collector import create_data_collector

# Import các module xử lý dữ liệu
try:
    from data_processors.cleaners.data_cleaner import DataCleaner
    from data_processors.cleaners.outlier_detector import OutlierDetector
    from data_processors.cleaners.missing_data_handler import MissingDataHandler
    from data_processors.feature_engineering.feature_generator import FeatureGenerator
    from data_processors.data_pipeline import DataPipeline
    DATA_PROCESSING_AVAILABLE = True
except ImportError:
    DATA_PROCESSING_AVAILABLE = False

# Thiết lập logger
logger = setup_component_logger("trading_system", "INFO")

class AutomatedTradingSystem:
    """
    Lớp quản lý chính của hệ thống giao dịch tự động.
    """
    
    def __init__(self):
        """
        Khởi tạo hệ thống giao dịch tự động.
        """
        logger.info("Khởi tạo Hệ thống Giao dịch Tự động")
        
        # Trạng thái hệ thống
        self.status = SystemStatus.INITIALIZING
        
        # Tải cấu hình hệ thống
        self.config = get_system_config()
        logger.info(f"Đã tải cấu hình hệ thống - Môi trường: {self.config.get('environment')}")
        
        # Kiểm tra biến môi trường bắt buộc
        missing_vars = env_manager.validate_required_vars()
        if missing_vars:
            logger.warning(f"Các biến môi trường bắt buộc chưa được thiết lập: {missing_vars}")
        
        # Khởi tạo các thành phần
        self.connectors = {}  # Lưu trữ các kết nối sàn giao dịch
        self.collectors = {}  # Lưu trữ các bộ thu thập dữ liệu
        
        # Kiểm tra tính khả dụng của các module
        if not DATA_PROCESSING_AVAILABLE:
            logger.warning("Các module xử lý dữ liệu không khả dụng. Vui lòng kiểm tra cài đặt.")
        
        # Khởi tạo các thành phần của hệ thống
        logger.info("Hệ thống đã sẵn sàng")
        self.status = SystemStatus.RUNNING
        
    # [Chuyển tất cả các phương thức khác từ AutomatedTradingSystem sang đây]
    # [Bao gồm init_exchange_connector, collect_historical_data, process_data, vv.]
    
    async def cleanup(self):
        """
        Dọn dẹp tài nguyên khi kết thúc.
        """
        logger.info("Dọn dẹp tài nguyên hệ thống")
        
        # Đóng các kết nối
        for name, connector in self.connectors.items():
            try:
                logger.debug(f"Đóng kết nối {name}")
                # Kiểm tra và gọi phương thức close nếu có
                if hasattr(connector, 'close'):
                    await connector.close()
            except Exception as e:
                logger.warning(f"Lỗi khi đóng kết nối {name}: {str(e)}")
        
        self.status = SystemStatus.STOPPED
        logger.info("Hệ thống đã dừng")