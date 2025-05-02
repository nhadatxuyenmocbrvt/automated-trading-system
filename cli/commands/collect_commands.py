"""
Xử lý lệnh thu thập dữ liệu.
File này cung cấp hàm xử lý lệnh 'collect' từ CLI.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd

from config.logging_config import get_logger
from config.env import get_env
from trading_system import AutomatedTradingSystem

# Thiết lập logger
logger = get_logger("collect_commands")

async def handle_collect_command(args, trading_system):
    """
    Xử lý lệnh collect từ CLI.
    
    Args:
        args: Đối tượng ArgumentParser đã parse
        trading_system: Instance của AutomatedTradingSystem
    """
    logger.info(f"Đang thu thập dữ liệu từ {args.exchange} cho {args.symbols}")
    
    # Lấy API key và secret
    api_key = args.api_key
    api_secret = args.api_secret
    
    if api_key is None or api_secret is None:
        # Thử lấy từ biến môi trường
        env = get_env()
        exchange_upper = args.exchange.upper()
        api_key = api_key or env.get(f"{exchange_upper}_API_KEY")
        api_secret = api_secret or env.get(f"{exchange_upper}_API_SECRET")
    
    # Thiết lập khoảng thời gian
    start_time = args.start_date
    end_time = args.end_date or datetime.now()
    days = args.days
    
    try:
        # Thu thập dữ liệu cho mỗi timeframe
        all_data = {}
        
        for timeframe in args.timeframes:
            logger.info(f"Thu thập dữ liệu khung thời gian {timeframe}")
            
            data = await trading_system.collect_data(
                exchange_id=args.exchange,
                symbols=args.symbols,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                days=days,
                is_futures=args.futures,
                api_key=api_key,
                api_secret=api_secret
            )
            
            if data:
                # Thêm vào all_data với key là timeframe
                for symbol, df in data.items():
                    all_data[f"{symbol}_{timeframe}"] = df
                
                # Log thông tin
                total_records = sum(len(df) for df in data.values())
                logger.info(f"Đã thu thập {total_records} dòng dữ liệu cho timeframe {timeframe}")
            else:
                logger.warning(f"Không thu thập được dữ liệu cho timeframe {timeframe}")
        
        # Xác định thư mục đầu ra
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = os.path.join(trading_system.data_dir, "raw")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(output_dir, exist_ok=True)
        
        # Lưu dữ liệu
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for key, df in all_data.items():
            symbol, timeframe = key.split("_", 1)
            symbol_safe = symbol.replace("/", "_").lower()
            
            file_path = os.path.join(output_dir, f"{symbol_safe}_{timeframe}_{timestamp}.csv")
            df.to_csv(file_path, index=False)
            
            logger.info(f"Đã lưu dữ liệu vào {file_path}")
        
        logger.info(f"Đã hoàn thành thu thập dữ liệu cho {len(args.symbols)} cặp giao dịch, {len(args.timeframes)} khung thời gian")
        
    except Exception as e:
        logger.error(f"Lỗi khi thu thập dữ liệu: {str(e)}")
        raise