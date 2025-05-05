# debug_trainer.py
import os
import sys
import logging
from pathlib import Path

# Thiết lập logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("debug")

logger.info("=== Bắt đầu kiểm tra imports ===")

try:
    from models.agents.dqn_agent import DQNAgent
    logger.info("✓ Import DQNAgent thành công")
except Exception as e:
    logger.error(f"✗ Lỗi import DQNAgent: {str(e)}")

try:
    from models.training_pipeline.trainer import Trainer
    logger.info("✓ Import Trainer thành công")
except Exception as e:
    logger.error(f"✗ Lỗi import Trainer: {str(e)}")

try:
    from environments.trading_gym.trading_env import TradingEnv
    logger.info("✓ Import TradingEnv thành công")
except Exception as e:
    logger.error(f"✗ Lỗi import TradingEnv: {str(e)}")

# Tạo instances
logger.info("=== Bắt đầu kiểm tra khởi tạo objects ===")

try:
    # Tìm file dữ liệu
    data_files = list(Path("data/processed").glob("*btc*usdt*.parquet"))
    if not data_files:
        logger.error("Không tìm thấy file dữ liệu")
        sys.exit(1)
    
    data_path = data_files[0]
    logger.info(f"Đã tìm thấy file dữ liệu: {data_path}")
    
    # Import các module cần thiết để tải dữ liệu
    from data_processors.data_pipeline import DataPipeline
    logger.info("✓ Import DataPipeline thành công")
    
    # Khởi tạo Data Pipeline
    pipeline = DataPipeline()
    logger.info("✓ Khởi tạo DataPipeline thành công")
    
    # Xác định định dạng file
    file_format = None
    if data_path.suffix == '.csv':
        file_format = 'csv'
    elif data_path.suffix == '.parquet':
        file_format = 'parquet'
    elif data_path.suffix == '.json':
        file_format = 'json'
        
    # Tải dữ liệu
    loaded_data = pipeline.load_data(
        file_paths=data_path,
        file_format=file_format
    )
    logger.info(f"✓ Tải dữ liệu thành công: {len(loaded_data)} symbols")
    
    # Lấy DataFrame đầu tiên
    first_symbol = next(iter(loaded_data.keys()))
    df = loaded_data[first_symbol]
    logger.info(f"✓ Lấy dữ liệu symbol {first_symbol} thành công, shape: {df.shape}")
    
    # Khởi tạo môi trường
    env = TradingEnv(
        data=df,
        symbol="BTC/USDT",
        timeframe="1h"
    )
    logger.info(f"✓ Khởi tạo TradingEnv thành công")
    logger.info(f"  - observation_space: {env.observation_space}")
    logger.info(f"  - action_space: {env.action_space}")
    
    # Khởi tạo DQN Agent
    agent_kwargs = {
        "state_dim": env.observation_space.shape,
        "action_dim": env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    }
    logger.info(f"  - state_dim: {agent_kwargs['state_dim']}")
    logger.info(f"  - action_dim: {agent_kwargs['action_dim']}")
    
    agent = DQNAgent(**agent_kwargs)
    logger.info(f"✓ Khởi tạo DQNAgent thành công")
    
    # Khởi tạo Trainer
    trainer = Trainer(
        agent=agent,
        env=env,
        output_dir="debug_model"
    )
    logger.info(f"✓ Khởi tạo Trainer thành công")
    
    # Kiểm tra các biến toàn cục trong trading_system.py
    import importlib
    spec = importlib.util.find_spec("trading_system")
    if spec is not None:
        ts_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ts_module)
        
        logger.info(f"Biến toàn cục trong trading_system.py:")
        logger.info(f"  - AGENTS_AVAILABLE: {getattr(ts_module, 'AGENTS_AVAILABLE', 'Không tìm thấy')}")
        logger.info(f"  - ENVIRONMENTS_AVAILABLE: {getattr(ts_module, 'ENVIRONMENTS_AVAILABLE', 'Không tìm thấy')}")
    
    logger.info("=== Kiểm tra hoàn tất thành công ===")
except Exception as e:
    logger.error(f"✗ Lỗi: {str(e)}", exc_info=True)