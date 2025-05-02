"""
Xử lý lệnh train.
File này cung cấp hàm xử lý lệnh 'train' từ CLI.
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
logger = get_logger("train_commands")

async def handle_train_command(args, trading_system):
    """
    Xử lý lệnh train từ CLI.
    
    Args:
        args: Đối tượng ArgumentParser đã parse
        trading_system: Instance của AutomatedTradingSystem
    """
    logger.info(f"Đang huấn luyện agent {args.agent}")
    
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
            window_size=args.window_size,
            reward_function=args.reward_function,
            include_positions=True,
            include_balance=True
        ):
            logger.error("Không thể thiết lập môi trường huấn luyện")
            return
        
        # Thiết lập agent
        agent_kwargs = {}
        if args.agent == "dqn":
            agent_kwargs = {
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.1,
                "epsilon_decay": 0.995,
                "batch_size": 64,
                "memory_size": 10000,
                "update_target_every": 100
            }
        elif args.agent == "ppo":
            agent_kwargs = {
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "clip_ratio": 0.2,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "lam": 0.95,
                "batch_size": 64,
                "epochs": 10
            }
        elif args.agent == "a2c":
            agent_kwargs = {
                "learning_rate": 0.0007,
                "gamma": 0.99,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "rms_prop_eps": 1e-5
            }
        
        if not trading_system.setup_agent(
            agent_type=args.agent,
            load_model=args.load_model,
            model_path=args.model_path,
            **agent_kwargs
        ):
            logger.error("Không thể thiết lập agent")
            return
        
        # Xác định đường dẫn lưu mô hình
        if args.save_path:
            save_path = args.save_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = os.path.join(trading_system.model_dir, args.agent)
            os.makedirs(model_dir, exist_ok=True)
            save_path = os.path.join(model_dir, f"{symbol_safe}_{timeframe}_{timestamp}.h5")
        
        # Thiết lập tham số huấn luyện
        train_kwargs = {
            "use_tensorboard": args.tensorboard,
            "render": args.render_training
        }
        
        # Huấn luyện agent
        results = trading_system.train_agent(
            episodes=args.episodes,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            save_path=save_path,
            **train_kwargs
        )
        
        if not results:
            logger.error("Huấn luyện không thành công")
            return
        
        # Hiển thị kết quả
        logger.info("Kết quả huấn luyện:")
        logger.info(f"  - Số episode đã huấn luyện: {args.episodes}")
        logger.info(f"  - Phần thưởng trung bình cuối: {results.get('final_avg_reward', 0):.4f}")
        logger.info(f"  - Phần thưởng tốt nhất: {results.get('best_reward', 0):.4f}")
        logger.info(f"  - Tỷ lệ thắng: {results.get('win_rate', 0)*100:.2f}%")
        
        # Lưu kết quả huấn luyện
        results_path = save_path.replace(".h5", "_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Đã lưu mô hình vào {save_path}")
        logger.info(f"Đã lưu kết quả huấn luyện vào {results_path}")
        logger.info(f"Đã hoàn thành huấn luyện agent {args.agent} cho {args.symbol}, timeframe {args.timeframe}")
        
    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện agent: {str(e)}")
        raise