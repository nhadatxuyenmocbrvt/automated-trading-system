"""
Xử lý lệnh trade.
File này cung cấp hàm xử lý lệnh 'trade' từ CLI.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
import logging
import json

from config.logging_config import get_logger
from config.env import get_env
from trading_system import AutomatedTradingSystem

# Thiết lập logger
logger = get_logger("trade_commands")

async def handle_trade_command(args, trading_system):
    """
    Xử lý lệnh trade từ CLI.
    
    Args:
        args: Đối tượng ArgumentParser đã parse
        trading_system: Instance của AutomatedTradingSystem
    """
    logger.info(f"Đang thiết lập giao dịch thực tế cho {args.symbol} trên {args.exchange}")
    
    # Kiểm tra chế độ
    if trading_system.mode == "production" and not args.test_mode:
        logger.warning("!!! CHẾ ĐỘ GIAO DỊCH THỰC TẾ !!!")
        logger.warning("Giao dịch sẽ được thực hiện trên tài khoản thật")
        
        # Yêu cầu xác nhận
        try:
            confirmation = input("Bạn có chắc chắn muốn giao dịch thực tế? (y/n): ")
            if confirmation.lower() != "y":
                logger.info("Đã hủy giao dịch thực tế")
                return
        except EOFError:
            # Chạy trong chế độ scripted, không có input
            logger.warning("Không thể yêu cầu xác nhận trong chế độ scripted, tiếp tục...")
    else:
        if args.test_mode:
            logger.info("Chạy ở CHẾ ĐỘ TEST - Không thực hiện giao dịch thực tế")
        else:
            logger.info(f"Chạy ở chế độ {trading_system.mode} - Không thực hiện giao dịch thực tế")
    
    # Lấy API key và secret
    api_key = args.api_key
    api_secret = args.api_secret
    
    if api_key is None or api_secret is None:
        # Thử lấy từ biến môi trường
        env = get_env()
        exchange_upper = args.exchange.upper()
        api_key = api_key or env.get(f"{exchange_upper}_API_KEY")
        api_secret = api_secret or env.get(f"{exchange_upper}_API_SECRET")
        
        if api_key is None or api_secret is None:
            logger.error("Thiếu API key hoặc secret. Vui lòng cung cấp qua tham số --api-key và --api-secret hoặc biến môi trường")
            return
    
    try:
        # Kiểm tra mô hình
        if not args.model_path or not os.path.exists(args.model_path):
            logger.error("Không tìm thấy file mô hình agent")
            return
        
        # Khởi tạo agent
        if not trading_system.setup_agent(
            agent_type=args.agent,
            load_model=True,
            model_path=args.model_path
        ):
            logger.error("Không thể thiết lập agent")
            return
        
        # Thiết lập trình thực thi giao dịch
        if not trading_system.setup_trade_executor(
            exchange_id=args.exchange,
            symbol=args.symbol,
            api_key=api_key,
            api_secret=api_secret,
            is_futures=args.futures,
            test_mode=args.test_mode,
            max_position_size=args.max_position_size
        ):
            logger.error("Không thể thiết lập trình thực thi giao dịch")
            return
        
        # Lưu cấu hình giao dịch
        config = {
            "exchange": args.exchange,
            "symbol": args.symbol,
            "agent": args.agent,
            "model_path": args.model_path,
            "is_futures": args.futures,
            "test_mode": args.test_mode,
            "max_trades": args.max_trades,
            "timeout": args.timeout,
            "stop_loss": args.stop_loss,
            "take_profit": args.take_profit,
            "max_position_size": args.max_position_size,
            "timestamp": datetime.now().isoformat()
        }
        
        config_dir = os.path.join(trading_system.data_dir, "trading_configs")
        os.makedirs(config_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = os.path.join(config_dir, f"trade_config_{timestamp}.json")
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Đã lưu cấu hình giao dịch vào {config_path}")
        
        # Bắt đầu giao dịch
        logger.info("Bắt đầu giao dịch...")
        
        results = await trading_system.start_trading(
            max_trades=args.max_trades,
            timeout=args.timeout,
            stop_loss_pct=args.stop_loss,
            take_profit_pct=args.take_profit
        )
        
        if not results:
            logger.error("Giao dịch không thành công")
            return
        
        # Hiển thị kết quả
        logger.info("Kết quả giao dịch:")
        logger.info(f"  - Số giao dịch: {results.get('total_trades', 0)}")
        logger.info(f"  - Lợi nhuận: {results.get('profit', 0):.4f}")
        logger.info(f"  - Tỷ lệ thắng: {results.get('win_rate', 0)*100:.2f}%")
        
        # Lưu kết quả giao dịch
        results_dir = os.path.join(trading_system.data_dir, "trading_results")
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, f"trade_results_{timestamp}.json")
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Đã lưu kết quả giao dịch vào {results_path}")
        logger.info("Đã hoàn thành phiên giao dịch")
        
    except KeyboardInterrupt:
        logger.info("Đã nhận lệnh thoát từ người dùng (Ctrl+C)")
        logger.info("Đang thoát an toàn...")
        # Thực hiện các thao tác cần thiết để thoát an toàn
        
    except Exception as e:
        logger.error(f"Lỗi khi giao dịch: {str(e)}")
        raise