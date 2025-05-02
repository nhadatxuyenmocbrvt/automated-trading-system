"""
Xử lý lệnh xử lý dữ liệu.
File này cung cấp hàm xử lý lệnh 'process' từ CLI.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd
import glob

from config.logging_config import get_logger
from trading_system import AutomatedTradingSystem

# Thiết lập logger
logger = get_logger("process_commands")

async def handle_process_command(args, trading_system):
    """
    Xử lý lệnh process từ CLI.
    
    Args:
        args: Đối tượng ArgumentParser đã parse
        trading_system: Instance của AutomatedTradingSystem
    """
    # Kiểm tra lệnh con
    if not hasattr(args, 'process_command') or args.process_command is None:
        logger.error("Thiếu lệnh con cho 'process'. Vui lòng chọn một trong các lệnh: clean, features, pipeline")
        return
    
    # Xử lý từng lệnh con
    if args.process_command == 'clean':
        await handle_clean_command(args, trading_system)
    elif args.process_command == 'features':
        await handle_features_command(args, trading_system)
    elif args.process_command == 'pipeline':
        await handle_pipeline_command(args, trading_system)
    else:
        logger.error(f"Lệnh con không hợp lệ: {args.process_command}")

async def handle_clean_command(args, trading_system):
    """
    Xử lý lệnh clean từ CLI.
    
    Args:
        args: Đối tượng ArgumentParser đã parse
        trading_system: Instance của AutomatedTradingSystem
    """
    logger.info(f"Đang làm sạch dữ liệu {args.data_type}")
    
    # Xác định thư mục đầu vào và đầu ra
    input_dir = args.input_dir or os.path.join(trading_system.data_dir, "raw")
    output_dir = args.output_dir or os.path.join(trading_system.data_dir, "cleaned")
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Tìm các file dữ liệu phù hợp
        input_files = []
        for symbol in args.symbols or ["*"]:
            symbol_safe = symbol.replace("/", "_").lower() if symbol != "*" else "*"
            
            if args.timeframes:
                for timeframe in args.timeframes:
                    pattern = os.path.join(input_dir, f"{symbol_safe}_{timeframe}_*.csv")
                    files = glob.glob(pattern)
                    input_files.extend(files)
            else:
                pattern = os.path.join(input_dir, f"{symbol_safe}_*.csv")
                files = glob.glob(pattern)
                input_files.extend(files)
        
        if not input_files:
            logger.warning(f"Không tìm thấy file dữ liệu trong {input_dir}")
            return
        
        logger.info(f"Tìm thấy {len(input_files)} file dữ liệu")
        
        # Tải dữ liệu
        data = trading_system.data_pipeline.load_data(input_files)
        
        if not data:
            logger.warning("Không tải được dữ liệu từ các file")
            return
        
        # Thiết lập các tham số làm sạch dữ liệu
        clean_ohlcv = args.data_type == "ohlcv"
        clean_orderbook = args.data_type == "orderbook"
        clean_trades = args.data_type == "trades"
        clean_sentiment = args.data_type == "sentiment"
        
        # Làm sạch dữ liệu
        cleaned_data = trading_system.data_pipeline.clean_data(
            data,
            clean_ohlcv=clean_ohlcv,
            clean_orderbook=clean_orderbook,
            clean_trades=clean_trades,
            clean_sentiment=clean_sentiment
        )
        
        # Lưu dữ liệu đã làm sạch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for symbol, df in cleaned_data.items():
            # Trích xuất thông tin từ tên symbol (nếu có timeframe)
            if "_" in symbol and symbol.split("_", 1)[1] in ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]:
                symbol_name, timeframe = symbol.split("_", 1)
                file_name = f"{symbol_name}_{timeframe}_cleaned_{timestamp}.csv"
            else:
                file_name = f"{symbol.replace('/', '_').lower()}_cleaned_{timestamp}.csv"
            
            file_path = os.path.join(output_dir, file_name)
            df.to_csv(file_path, index=False)
            
            logger.info(f"Đã lưu dữ liệu đã làm sạch vào {file_path}")
        
        logger.info(f"Đã hoàn thành làm sạch dữ liệu cho {len(cleaned_data)} cặp giao dịch")
        
    except Exception as e:
        logger.error(f"Lỗi khi làm sạch dữ liệu: {str(e)}")
        raise

async def handle_features_command(args, trading_system):
    """
    Xử lý lệnh features từ CLI.
    
    Args:
        args: Đối tượng ArgumentParser đã parse
        trading_system: Instance của AutomatedTradingSystem
    """
    logger.info(f"Đang tạo đặc trưng kỹ thuật cho dữ liệu {args.data_type}")
    
    # Xác định thư mục đầu vào và đầu ra
    input_dir = args.input_dir or os.path.join(trading_system.data_dir, "cleaned")
    output_dir = args.output_dir or os.path.join(trading_system.data_dir, "features")
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Tìm các file dữ liệu phù hợp
        input_files = []
        for symbol in args.symbols or ["*"]:
            symbol_safe = symbol.replace("/", "_").lower() if symbol != "*" else "*"
            
            if args.timeframes:
                for timeframe in args.timeframes:
                    pattern = os.path.join(input_dir, f"{symbol_safe}_{timeframe}*_cleaned_*.csv")
                    files = glob.glob(pattern)
                    input_files.extend(files)
            else:
                pattern = os.path.join(input_dir, f"{symbol_safe}*_cleaned_*.csv")
                files = glob.glob(pattern)
                input_files.extend(files)
        
        if not input_files:
            logger.warning(f"Không tìm thấy file dữ liệu đã làm sạch trong {input_dir}")
            return
        
        logger.info(f"Tìm thấy {len(input_files)} file dữ liệu đã làm sạch")
        
        # Tải dữ liệu
        data = trading_system.data_pipeline.load_data(input_files)
        
        if not data:
            logger.warning("Không tải được dữ liệu từ các file")
            return
        
        # Thiết lập cấu hình đặc trưng
        feature_configs = {}
        for symbol in data.keys():
            if args.all_indicators:
                # Sử dụng tất cả các chỉ báo
                feature_configs[symbol] = {
                    "feature_names": None,  # Sử dụng tất cả
                    "preprocessor_names": ["normalize"],
                    "transformer_names": None,  # Sử dụng mặc định
                    "feature_selector": "statistical_correlation"
                }
            elif args.indicators:
                # Sử dụng các chỉ báo được chỉ định
                feature_configs[symbol] = {
                    "feature_names": args.indicators,
                    "preprocessor_names": ["normalize"],
                    "transformer_names": None,  # Sử dụng mặc định
                    "feature_selector": "statistical_correlation"
                }
            else:
                # Sử dụng các chỉ báo phổ biến mặc định
                feature_configs[symbol] = {
                    "feature_names": [
                        "trend_sma", "trend_ema", "trend_macd", "trend_adx",
                        "momentum_rsi", "momentum_stoch", "momentum_cci",
                        "volatility_bbands", "volatility_atr",
                        "volume_obv", "volume_vwap"
                    ],
                    "preprocessor_names": ["normalize"],
                    "transformer_names": None,  # Sử dụng mặc định
                    "feature_selector": "statistical_correlation"
                }
        
        # Tạo đặc trưng
        featured_data = trading_system.data_pipeline.generate_features(
            data,
            feature_configs=feature_configs
        )
        
        if not featured_data:
            logger.warning("Không tạo được đặc trưng từ dữ liệu")
            return
        
        # Lưu dữ liệu đã tạo đặc trưng
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for symbol, df in featured_data.items():
            # Trích xuất thông tin từ tên symbol (nếu có timeframe)
            if "_" in symbol and symbol.split("_", 1)[1] in ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]:
                symbol_name, timeframe = symbol.split("_", 1)
                file_name = f"{symbol_name}_{timeframe}_featured_{timestamp}.csv"
            else:
                file_name = f"{symbol.replace('/', '_').lower()}_featured_{timestamp}.csv"
            
            file_path = os.path.join(output_dir, file_name)
            df.to_csv(file_path, index=False)
            
            logger.info(f"Đã lưu dữ liệu đã tạo đặc trưng vào {file_path} ({len(df.columns)} đặc trưng)")
        
        logger.info(f"Đã hoàn thành tạo đặc trưng cho {len(featured_data)} cặp giao dịch")
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo đặc trưng: {str(e)}")
        raise

async def handle_pipeline_command(args, trading_system):
    """
    Xử lý lệnh pipeline từ CLI.
    
    Args:
        args: Đối tượng ArgumentParser đã parse
        trading_system: Instance của AutomatedTradingSystem
    """
    logger.info("Đang chạy pipeline xử lý dữ liệu")
    
    # Xác định thư mục đầu ra
    output_dir = args.output_dir or os.path.join(trading_system.data_dir, "processed")
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Xác định nguồn dữ liệu
        if args.input_files:
            # Sử dụng các file đầu vào
            input_files = args.input_files
            exchange_id = None
            symbols = None
            timeframe = None
            start_time = None
            end_time = None
            
            logger.info(f"Sử dụng dữ liệu từ {len(input_files)} file đầu vào")
            
        elif args.exchange and args.symbols:
            # Thu thập dữ liệu mới
            exchange_id = args.exchange
            symbols = args.symbols
            timeframe = args.timeframes[0] if args.timeframes else "1h"
            start_time = args.start_date
            end_time = args.end_date
            input_files = None
            
            logger.info(f"Thu thập dữ liệu mới từ {exchange_id} cho {symbols}, timeframe {timeframe}")
            
        else:
            logger.error("Thiếu thông tin đầu vào: cần cung cấp --input-files hoặc (--exchange và --symbols)")
            return
        
        # Thiết lập pipeline
        steps = [
            {"name": "collect_data", "enabled": exchange_id is not None},
            {"name": "load_data", "enabled": input_files is not None},
            {"name": "clean_data", "enabled": not args.skip_clean},
            {"name": "generate_features", "enabled": not args.skip_features},
            {"name": "save_data", "enabled": True}
        ]
        
        pipeline_name = args.pipeline_name or f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Đăng ký pipeline
        trading_system.data_pipeline.register_pipeline(
            name=pipeline_name,
            steps=steps,
            description="Pipeline xử lý dữ liệu từ CLI"
        )
        
        # Chạy pipeline
        processed_data = await trading_system.data_pipeline.run_pipeline(
            pipeline_name=pipeline_name,
            input_files=input_files,
            exchange_id=exchange_id,
            symbols=symbols,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            output_dir=output_dir
        )
        
        if not processed_data:
            logger.warning("Không có dữ liệu sau khi xử lý")
            return
        
        logger.info(f"Đã hoàn thành pipeline xử lý dữ liệu cho {len(processed_data)} cặp giao dịch")
        
        # In thông tin về dữ liệu đã xử lý
        for symbol, df in processed_data.items():
            logger.info(f"  - {symbol}: {len(df)} dòng, {len(df.columns)} cột")
        
    except Exception as e:
        logger.error(f"Lỗi khi chạy pipeline xử lý dữ liệu: {str(e)}")
        raise