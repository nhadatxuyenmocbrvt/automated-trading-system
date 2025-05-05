"""
Xử lý lệnh huấn luyện agent.
File này định nghĩa các tham số và xử lý cho lệnh 'train' trên CLI.
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

def setup_train_parser(subparsers) -> None:
    """
    Thiết lập parser cho lệnh 'train'.
    
    Args:
        subparsers: Subparsers object từ argparse
    """
    train_parser = subparsers.add_parser(
        'train',
        help='Huấn luyện agent giao dịch',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Loại agent
    train_parser.add_argument(
        '--agent', 
        type=str, 
        choices=['dqn', 'ppo', 'a2c'], 
        default='dqn',
        help='Loại agent cần huấn luyện'
    )
    
    # Tham số dữ liệu
    train_parser.add_argument(
        '--data-path', 
        type=str,
        help='Đường dẫn file dữ liệu huấn luyện'
    )
    
    train_parser.add_argument(
        '--symbol', 
        type=str, 
        default='BTC/USDT',
        help='Cặp giao dịch huấn luyện'
    )
    
    train_parser.add_argument(
        '--timeframe', 
        type=str, 
        default='1h',
        help='Khung thời gian huấn luyện'
    )
    
    # Tham số huấn luyện
    train_parser.add_argument(
        '--episodes', 
        type=int, 
        default=1000,
        help='Số episode huấn luyện'
    )
    
    train_parser.add_argument(
        '--initial-balance', 
        type=float, 
        default=10000.0,
        help='Số dư ban đầu cho môi trường'
    )
    
    train_parser.add_argument(
        '--leverage', 
        type=float, 
        default=1.0,
        help='Đòn bẩy giao dịch'
    )
    
    train_parser.add_argument(
        '--fee-rate', 
        type=float, 
        default=0.001,
        help='Tỷ lệ phí giao dịch (0.001 = 0.1%)'
    )
    
    train_parser.add_argument(
        '--reward-function', 
        type=str, 
        choices=['profit', 'risk_adjusted', 'sharpe', 'sortino', 'calmar', 'custom'], 
        default='profit',
        help='Hàm phần thưởng'
    )
    
    # Tham số mô hình
    train_parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=0.001,
        help='Tốc độ học'
    )
    
    train_parser.add_argument(
        '--gamma', 
        type=float, 
        default=0.99,
        help='Hệ số giảm phần thưởng'
    )
    
    train_parser.add_argument(
        '--batch-size', 
        type=int, 
        default=64,
        help='Kích thước batch huấn luyện'
    )
    
    train_parser.add_argument(
        '--epsilon', 
        type=float, 
        default=1.0,
        help='Giá trị epsilon ban đầu cho exploration'
    )
    
    train_parser.add_argument(
        '--epsilon-decay', 
        type=float, 
        default=0.995,
        help='Tốc độ giảm epsilon'
    )
    
    train_parser.add_argument(
        '--epsilon-min', 
        type=float, 
        default=0.01,
        help='Giá trị epsilon tối thiểu'
    )
    # Đường dẫn đầu ra
    train_parser.add_argument(
        '--output-dir', 
        type=str,
        help='Thư mục lưu mô hình'
    )
    train_parser.add_argument(
    '--continue-training', 
    action='store_true',
    help='Tiếp tục huấn luyện từ mô hình đã lưu'
    )    
    
    train_parser.set_defaults(func=handle_train_command)

def handle_train_command(args: argparse.Namespace, system: AutomatedTradingSystem) -> int:
    """
    Xử lý lệnh 'train'.
    
    Args:
        args: Các tham số dòng lệnh
        system: Instance của AutomatedTradingSystem
        
    Returns:
        int: Mã kết quả (0 = thành công)
    """
    logger = get_logger('train_command')
    
    try:
        # Xử lý tham số đầu vào
        agent_type = args.agent
        data_path = args.data_path
        symbol = args.symbol
        timeframe = args.timeframe
        episodes = args.episodes
        output_dir = args.output_dir
        
        # Nếu không có data_path, tìm trong thư mục mặc định
        if not data_path:
            # Phần tìm kiếm dữ liệu giữ nguyên như cũ...
            # Sử dụng đường dẫn tương đối tới thư mục processed
            default_dir = system.data_dir / "processed"
            
            # Kiểm tra xem thư mục này có tồn tại không
            if not default_dir.exists():
                default_dir = system.data_dir / "collected"
            
            # Tìm tất cả các file .parquet trong thư mục
            symbol_safe = symbol.replace('/', '_').lower()
            
            # Thử tìm file phù hợp với symbol và timeframe
            pattern = f"*{symbol_safe}*{timeframe}*.parquet"
            matching_files = list(default_dir.glob(pattern))
            
            # Nếu không tìm thấy, thử tìm chỉ với symbol
            if not matching_files:
                pattern = f"*{symbol_safe}*.parquet"
                matching_files = list(default_dir.glob(pattern))
            
            # Nếu vẫn không tìm thấy, thử tìm tất cả các file .parquet
            if not matching_files:
                pattern = "*.parquet"
                matching_files = list(default_dir.glob(pattern))
            
            if matching_files:
                # Sắp xếp theo thời gian sửa đổi để lấy file mới nhất
                matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                data_path = str(matching_files[0])
                logger.info(f"Tự động tìm thấy file dữ liệu: {data_path}")
            else:
                logger.error(f"Không tìm thấy file dữ liệu nào cho {symbol}, {timeframe}")
                return 1
        
        # Chuẩn bị tham số cho môi trường
        env_kwargs = {
            "initial_balance": args.initial_balance,
            "leverage": args.leverage,
            "fee_rate": args.fee_rate,
            "reward_function": args.reward_function
        }
        
        # Chuẩn bị tham số cho agent
        agent_kwargs = {
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "batch_size": args.batch_size,
            "epsilon": args.epsilon,
            "epsilon_decay": args.epsilon_decay,
            "epsilon_min": args.epsilon_min
        }
        
        # Kết hợp tham số
        train_kwargs = {
            "env_kwargs": env_kwargs,
            "agent_kwargs": agent_kwargs
        }
        
        # Chuyển đổi thành đường dẫn nếu có
        if output_dir:
            output_dir = Path(output_dir)
        
        # Xử lý tham số tiếp tục huấn luyện
        if hasattr(args, 'continue_training') and args.continue_training:
            # Xác định đường dẫn mô hình dựa trên loại agent, symbol và timeframe
            if output_dir is None:
                # Sử dụng đường dẫn mặc định nếu không chỉ định
                model_dir = system.model_dir / agent_type / symbol.replace('/', '_') / timeframe
            else:
                # Sử dụng đường dẫn được chỉ định
                model_dir = output_dir / agent_type / symbol.replace('/', '_') / timeframe
            
            # Kiểm tra xem mô hình tốt nhất có tồn tại không
            best_model_path = model_dir / "models" / "best_model"
            
            if best_model_path.exists():
                # Nếu có, sử dụng mô hình tốt nhất
                train_kwargs["continue_training"] = True
                train_kwargs["model_path"] = str(best_model_path)
                logger.info(f"Tiếp tục huấn luyện từ mô hình tốt nhất: {best_model_path}")
            else:
                # Nếu không, tìm checkpoint gần nhất
                checkpoint_dir = model_dir / "checkpoints"
                
                if checkpoint_dir.exists():
                    # Tìm tất cả các checkpoints
                    checkpoints = list(checkpoint_dir.glob("checkpoint_episode_*"))
                    
                    if checkpoints:
                        # Sắp xếp theo số episode giảm dần
                        try:
                            checkpoints.sort(key=lambda x: int(x.name.split("_")[-1]), reverse=True)
                            latest_checkpoint = checkpoints[0]
                            
                            train_kwargs["continue_training"] = True
                            train_kwargs["checkpoint"] = str(latest_checkpoint)
                            logger.info(f"Tiếp tục huấn luyện từ checkpoint: {latest_checkpoint}")
                        except Exception as e:
                            logger.warning(f"Không thể sắp xếp checkpoints: {str(e)}")
                            logger.warning("Bắt đầu huấn luyện mới")
                    else:
                        logger.warning("Không tìm thấy checkpoint, bắt đầu huấn luyện mới")
                else:
                    logger.warning("Không tìm thấy thư mục checkpoints, bắt đầu huấn luyện mới")
        
        # Thực hiện huấn luyện
        logger.info(f"Bắt đầu huấn luyện agent {agent_type} trên {symbol}, {timeframe}")
        
        success, model_path = asyncio.run(system.train_agent(
            data_path=data_path,
            agent_type=agent_type,
            symbol=symbol,
            timeframe=timeframe,
            num_episodes=episodes,
            output_dir=output_dir,
            **train_kwargs
        ))
        
        if success:
            logger.info(f"Huấn luyện thành công! Mô hình được lưu tại: {model_path}")
            
            # Đánh giá agent
            logger.info("Đánh giá agent trên tập kiểm thử...")
            eval_results = system.evaluate_agent(
                model_path=model_path,
                data_path=data_path,
                agent_type=agent_type,
                symbol=symbol,
                timeframe=timeframe,
                num_episodes=10,
                **train_kwargs
            )
            
            if eval_results:
                mean_reward = eval_results.get("mean_reward", 0)
                std_reward = eval_results.get("std_reward", 0)
                min_reward = eval_results.get("min_reward", 0)
                max_reward = eval_results.get("max_reward", 0)
                
                logger.info(
                    f"Kết quả đánh giá: Mean reward: {mean_reward:.2f} ± {std_reward:.2f}, "
                    f"Min: {min_reward:.2f}, Max: {max_reward:.2f}"
                )
            
            return 0
        else:
            logger.error("Huấn luyện thất bại")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Đã hủy huấn luyện")
        return 130
    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện agent: {str(e)}", exc_info=True)
        return 1