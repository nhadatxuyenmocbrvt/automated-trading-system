"""
Xử lý lệnh backtest.
File này cung cấp hàm xử lý lệnh 'backtest' từ CLI.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd
import glob
import json

from config.logging_config import get_logger
from trading_system import AutomatedTradingSystem

# Thiết lập logger
logger = get_logger("backtest_commands")

async def handle_backtest_command(args, trading_system):
    """
    Xử lý lệnh backtest từ CLI.
    
    Args:
        args: Đối tượng ArgumentParser đã parse
        trading_system: Instance của AutomatedTradingSystem
    """
    logger.info(f"Đang thực hiện backtest cho agent {args.agent}")
    
    # Xác định thư mục dữ liệu
    data_dir = args.data_dir or trading_system.data_dir
    
    # Tìm file dữ liệu phù hợp
    symbol_safe = args.symbol.replace("/", "_").lower()
    timeframe = args.timeframe
    
    pattern = os.path.join(data_dir, "processed", f"{symbol_safe}_{timeframe}*_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        # Thử tìm trong thư mục features
        pattern = os.path.join(data_dir, "features", f"{symbol_safe}_{timeframe}*_featured_*.csv")
        files = glob.glob(pattern)
    
    if not files:
        # Thử tìm trong thư mục cleaned
        pattern = os.path.join(data_dir, "cleaned", f"{symbol_safe}_{timeframe}*_cleaned_*.csv")
        files = glob.glob(pattern)
    
    if not files:
        # Thử tìm trong thư mục raw
        pattern = os.path.join(data_dir, "raw", f"{symbol_safe}_{timeframe}*.csv")
        files = glob.glob(pattern)
    
    if not files:
        logger.error(f"Không tìm thấy dữ liệu cho {args.symbol} với timeframe {args.timeframe}")
        return
    
    # Sử dụng file mới nhất
    newest_file = max(files, key=os.path.getctime)
    logger.info(f"Sử dụng dữ liệu từ {newest_file}")
    
    try:
        # Tải dữ liệu
        df = pd.read_csv(newest_file)
        
        # Chuyển cột timestamp sang datetime nếu có
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        data = {args.symbol: df}
        
        # Thiết lập môi trường
        if not trading_system.setup_environment(
            data=data,
            symbol=args.symbol,
            initial_balance=args.initial_balance,
            max_positions=args.max_positions,
            window_size=100,
            reward_function="profit"
        ):
            logger.error("Không thể thiết lập môi trường backtest")
            return
        
        # Thiết lập agent
        if not trading_system.setup_agent(
            agent_type=args.agent,
            load_model=args.model_path is not None,
            model_path=args.model_path
        ):
            logger.error("Không thể thiết lập agent")
            return
        
        # Xác định đường dẫn đầu ra
        if args.output_path:
            output_path = args.output_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(trading_system.data_dir, "backtest_results")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"backtest_{symbol_safe}_{timeframe}_{args.agent}_{timestamp}.csv")
        
        # Chạy backtest
        results = trading_system.backtest(
            render_mode=args.render_mode,
            output_path=output_path
        )
        
        if not results:
            logger.error("Backtest không thành công")
            return
        
        # Hiển thị kết quả
        logger.info("Kết quả backtest:")
        logger.info(f"  - Số dư ban đầu: {args.initial_balance:.2f}")
        logger.info(f"  - Số dư cuối: {results['final_balance']:.2f}")
        logger.info(f"  - NAV cuối: {results['final_nav']:.2f}")
        logger.info(f"  - Tỷ suất lợi nhuận: {results['return_pct']:.2f}%")
        logger.info(f"  - Tỷ lệ thắng: {results['win_rate']*100:.2f}%")
        logger.info(f"  - Drawdown tối đa: {results['max_drawdown']*100:.2f}%")
        logger.info(f"  - Số giao dịch: {results['trade_count']}")
        
        # Lưu thêm file kết quả JSON
        json_output = output_path.replace(".csv", ".json")
        with open(json_output, "w") as f:
            # Chuyển các timestamp thành chuỗi để có thể lưu JSON
            if "history" in results and "timestamps" in results["history"]:
                results["history"]["timestamps"] = [str(ts) for ts in results["history"]["timestamps"]]
            
            json.dump(results, f, indent=4)
        
        logger.info(f"Đã lưu kết quả chi tiết vào {json_output}")
        logger.info(f"Đã hoàn thành backtest cho {args.symbol}, timeframe {args.timeframe}")
        
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện backtest: {str(e)}")
        raise