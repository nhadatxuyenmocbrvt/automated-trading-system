#!/usr/bin/env python3
"""
Điểm khởi chạy chính của Hệ thống Giao dịch Tự động.
File này cung cấp giao diện dòng lệnh để khởi chạy các thành phần
khác nhau của hệ thống, bao gồm thu thập dữ liệu, huấn luyện agent,
backtest, và triển khai giao dịch thực tế.
"""

import os
import sys
import argparse
import logging
import asyncio
from pathlib import Path
import datetime
from typing import Dict, List, Optional, Any

# Thêm thư mục gốc vào sys.path để import các module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import các module của hệ thống
from config.system_config import get_system_config, system_config
from config.logging_config import get_logger, setup_component_logger
from config.env import get_env, env_manager
from config.constants import SystemStatus, Exchange

# Import các module thu thập dữ liệu
from data_collectors.exchange_api.binance_connector import BinanceConnector
from data_collectors.exchange_api.generic_connector import ExchangeConnector
from data_collectors.market_data.historical_data_collector import create_data_collector

# Thiết lập logger cho main
logger = setup_component_logger("main", "INFO")

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
        
        # Khởi tạo các thành phần của hệ thống
        logger.info("Hệ thống đã sẵn sàng")
        self.status = SystemStatus.RUNNING
    
    async def init_exchange_connector(self, exchange_id: str, is_futures: bool = False, 
                                     testnet: bool = True) -> ExchangeConnector:
        """
        Khởi tạo kết nối với sàn giao dịch.
        
        Args:
            exchange_id: ID của sàn giao dịch
            is_futures: True để sử dụng thị trường futures
            testnet: True để sử dụng môi trường test
            
        Returns:
            Đối tượng kết nối với sàn giao dịch
        """
        try:
            connector_key = f"{exchange_id}_{'futures' if is_futures else 'spot'}"
            
            # Kiểm tra xem connector đã tồn tại chưa
            if connector_key in self.connectors:
                logger.info(f"Sử dụng kết nối {connector_key} hiện có")
                return self.connectors[connector_key]
            
            # Lấy API key và secret từ biến môi trường
            api_key = get_env(f"{exchange_id.upper()}_API_KEY", "")
            api_secret = get_env(f"{exchange_id.upper()}_API_SECRET", "")
            
            # Tạo connector phù hợp với sàn giao dịch
            if exchange_id.lower() == "binance":
                connector = BinanceConnector(
                    api_key=api_key,
                    api_secret=api_secret,
                    is_futures=is_futures,
                    testnet=testnet
                )
                logger.info(f"Đã tạo kết nối Binance {'Futures' if is_futures else 'Spot'}")
            else:
                # Sử dụng generic connector cho các sàn khác
                connector = ExchangeConnector(
                    exchange_id=exchange_id,
                    api_key=api_key,
                    api_secret=api_secret,
                    testnet=testnet
                )
                logger.info(f"Đã tạo kết nối {exchange_id}")
            
            # Lưu vào cache
            self.connectors[connector_key] = connector
            return connector
            
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo kết nối {exchange_id}: {str(e)}")
            raise
    
    async def collect_historical_data(self, exchange_id: str, symbols: List[str], 
                                     timeframes: List[str], days_back: int = 30,
                                     is_futures: bool = False) -> None:
        """
        Thu thập dữ liệu lịch sử từ sàn giao dịch.
        
        Args:
            exchange_id: ID của sàn giao dịch
            symbols: Danh sách cặp giao dịch
            timeframes: Danh sách timeframe
            days_back: Số ngày lấy dữ liệu
            is_futures: True để sử dụng thị trường futures
        """
        try:
            # Khởi tạo connector
            connector = await self.init_exchange_connector(
                exchange_id=exchange_id,
                is_futures=is_futures,
                testnet=False  # Sử dụng mainnet để lấy dữ liệu thực
            )
            
            # Khởi tạo collector
            collector = await create_data_collector(
                exchange_id=exchange_id,
                api_key=connector.api_key,
                api_secret=connector.api_secret,
                sandbox=False,  # Sử dụng mainnet để lấy dữ liệu thực
                is_futures=is_futures
            )
            
            # Lưu vào cache
            collector_key = f"{exchange_id}_{'futures' if is_futures else 'spot'}"
            self.collectors[collector_key] = collector
            
            # Thời gian bắt đầu và kết thúc
            end_time = datetime.datetime.now()
            start_time = end_time - datetime.timedelta(days=days_back)
            
            logger.info(f"Bắt đầu thu thập dữ liệu từ {start_time} đến {end_time}")
            logger.info(f"Cặp giao dịch: {symbols}")
            logger.info(f"Timeframes: {timeframes}")
            
            # Thu thập dữ liệu cho mỗi cặp và timeframe
            for symbol in symbols:
                for timeframe in timeframes:
                    logger.info(f"Thu thập dữ liệu {symbol} - {timeframe}")
                    
                    try:
                        # Thu thập OHLCV
                        df = await collector.collect_ohlcv(
                            symbol=symbol,
                            timeframe=timeframe,
                            start_time=start_time,
                            end_time=end_time
                        )
                        
                        logger.info(f"Đã thu thập {len(df) if df is not None else 0} bản ghi cho {symbol} - {timeframe}")
                        
                    except Exception as e:
                        logger.error(f"Lỗi khi thu thập dữ liệu {symbol} - {timeframe}: {str(e)}")
            
            logger.info(f"Đã hoàn thành thu thập dữ liệu lịch sử cho {exchange_id}")
            
        except Exception as e:
            logger.error(f"Lỗi khi thu thập dữ liệu lịch sử: {str(e)}")
            raise

    async def run_backtest(self):
        """
        Chạy backtest trên dữ liệu lịch sử.
        """
        logger.info("Chức năng backtest chưa được triển khai")
        # TODO: Triển khai khi module backtest đã sẵn sàng
    
    async def train_agent(self):
        """
        Huấn luyện agent giao dịch.
        """
        logger.info("Chức năng huấn luyện agent chưa được triển khai")
        # TODO: Triển khai khi module models đã sẵn sàng
    
    async def start_trading(self):
        """
        Bắt đầu giao dịch thực tế.
        """
        logger.info("Chức năng giao dịch thực tế chưa được triển khai")
        # TODO: Triển khai khi module deployment đã sẵn sàng
    
    async def start_dashboard(self):
        """
        Khởi chạy dashboard.
        """
        logger.info("Chức năng dashboard chưa được triển khai")
        # TODO: Triển khai khi module streamlit_dashboard đã sẵn sàng

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

async def main():
    """
    Hàm main chính của hệ thống.
    """
    # Tạo parser cho command-line arguments
    parser = argparse.ArgumentParser(description='Hệ thống Giao dịch Tự động')
    subparsers = parser.add_subparsers(dest='command', help='Lệnh cần thực hiện')
    
    # Tạo subparser cho thu thập dữ liệu
    collect_parser = subparsers.add_parser('collect', help='Thu thập dữ liệu lịch sử')
    collect_parser.add_argument('--exchange', type=str, default='binance', help='Tên sàn giao dịch')
    collect_parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDT', 'ETH/USDT'], 
                               help='Danh sách cặp giao dịch')
    collect_parser.add_argument('--timeframes', type=str, nargs='+', default=['1h', '4h', '1d'], 
                               help='Danh sách timeframe')
    collect_parser.add_argument('--days', type=int, default=30, help='Số ngày lấy dữ liệu')
    collect_parser.add_argument('--futures', action='store_true', help='Sử dụng thị trường futures')
    
    # Tạo subparser cho backtest
    backtest_parser = subparsers.add_parser('backtest', help='Chạy backtest')
    backtest_parser.add_argument('--strategy', type=str, default='dqn', help='Chiến lược giao dịch')
    
    # Tạo subparser cho huấn luyện
    train_parser = subparsers.add_parser('train', help='Huấn luyện agent')
    train_parser.add_argument('--agent', type=str, default='dqn', help='Loại agent')
    
    # Tạo subparser cho giao dịch thực tế
    trade_parser = subparsers.add_parser('trade', help='Giao dịch thực tế')
    trade_parser.add_argument('--exchange', type=str, default='binance', help='Tên sàn giao dịch')
    trade_parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDT'], 
                             help='Danh sách cặp giao dịch')
    
    # Tạo subparser cho dashboard
    dashboard_parser = subparsers.add_parser('dashboard', help='Khởi chạy dashboard')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Nếu không có lệnh nào được cung cấp, hiển thị help
    if not args.command:
        parser.print_help()
        return
    
    # Khởi tạo hệ thống
    system = AutomatedTradingSystem()
    
    try:
        # Thực hiện lệnh tương ứng
        if args.command == 'collect':
            await system.collect_historical_data(
                exchange_id=args.exchange,
                symbols=args.symbols,
                timeframes=args.timeframes,
                days_back=args.days,
                is_futures=args.futures
            )
        elif args.command == 'backtest':
            await system.run_backtest()
        elif args.command == 'train':
            await system.train_agent()
        elif args.command == 'trade':
            await system.start_trading()
        elif args.command == 'dashboard':
            await system.start_dashboard()
    except KeyboardInterrupt:
        logger.info("Nhận tín hiệu dừng từ người dùng")
    except Exception as e:
        logger.error(f"Lỗi không xử lý được: {str(e)}", exc_info=True)
    finally:
        # Dọn dẹp tài nguyên
        await system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())