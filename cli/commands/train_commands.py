"""
CLI commands cho huấn luyện agent.
File này cung cấp các lệnh để huấn luyện agent, tối ưu hóa siêu tham số,
tiếp tục huấn luyện từ checkpoint, và thực hiện các tác vụ liên quan đến huấn luyện.
"""

import os
import sys
import click
import json
import logging
import numpy as np
import pandas as pd
import importlib
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import get_logger, setup_component_logger
from config.system_config import get_system_config
from config.constants import AgentType, Exchange, Timeframe, EXCHANGE_TIMEFRAMES
from config.utils.validators import is_valid_timeframe, is_valid_trading_pair, validate_config

from models.training_pipeline.trainer import Trainer
from models.training_pipeline.hyperparameter_tuner import HyperparameterTuner
from environments.trading_gym.trading_env import TradingEnv
from automation.metrics.performance_metrics import AutomationPerformanceMetrics

# Thiết lập logger
logger = get_logger("train_commands")

# Lấy cấu hình hệ thống
system_config = get_system_config()

# Thư mục gốc của dự án
BASE_DIR = Path(__file__).parent.parent.parent

# Mặc định cho thư mục dữ liệu
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "saved_models"

# Định nghĩa nhóm lệnh train
@click.group(name="train")
def train_commands():
    """Các lệnh cho huấn luyện và quản lý agent."""
    pass

@train_commands.command(name="agent")
@click.argument("agent_type", type=click.Choice([agent.value for agent in AgentType]))
@click.option("--data-path", "-d", type=click.Path(exists=True), help="Đường dẫn đến file dữ liệu huấn luyện")
@click.option("--symbol", "-s", type=str, default="BTC/USDT", help="Cặp giao dịch (ví dụ: BTC/USDT)")
@click.option("--timeframe", "-t", type=str, default="1h", help="Khung thời gian (1m, 5m, 15m, 1h, 4h, 1d, ...)")
@click.option("--episodes", "-e", type=int, default=1000, help="Số lượng episodes huấn luyện")
@click.option("--output-dir", "-o", type=click.Path(), default=str(DEFAULT_OUTPUT_DIR), help="Thư mục đầu ra cho kết quả huấn luyện")
@click.option("--config-file", "-c", type=click.Path(exists=True), help="File cấu hình cho huấn luyện (JSON)")
@click.option("--random-seed", "-r", type=int, default=42, help="Seed cho bộ sinh số ngẫu nhiên")
@click.option("--leverage", "-l", type=float, default=1.0, help="Đòn bẩy")
@click.option("--initial-balance", "-b", type=float, default=10000.0, help="Số dư ban đầu")
@click.option("--window-size", "-w", type=int, default=100, help="Kích thước cửa sổ dữ liệu")
@click.option("--eval-interval", "-i", type=int, default=10, help="Khoảng thời gian đánh giá (episodes)")
@click.option("--save-interval", "-v", type=int, default=100, help="Khoảng thời gian lưu mô hình (episodes)")
@click.option("--render/--no-render", default=False, help="Có hiển thị quá trình huấn luyện không")
@click.option("--gpu/--cpu", default=False, help="Sử dụng GPU hay không")
@click.option("--verbose", type=click.Choice(["0", "1", "2"]), default="1", help="Mức độ chi tiết (0: im lặng, 1: bình thường, 2: chi tiết)")
@click.option("--experiment-name", "-n", type=str, help="Tên thí nghiệm")
@click.option("--tensorboard/--no-tensorboard", default=True, help="Sử dụng TensorBoard")
@click.option("--early-stopping/--no-early-stopping", default=True, help="Sử dụng early stopping")
def train_agent(
    agent_type: str,
    data_path: Optional[str],
    symbol: str,
    timeframe: str,
    episodes: int,
    output_dir: str,
    config_file: Optional[str],
    random_seed: int,
    leverage: float,
    initial_balance: float,
    window_size: int,
    eval_interval: int,
    save_interval: int,
    render: bool,
    gpu: bool,
    verbose: str,
    experiment_name: Optional[str],
    tensorboard: bool,
    early_stopping: bool
):
    """
    Huấn luyện agent giao dịch mới.
    
    Lệnh này khởi tạo và huấn luyện một agent mới dựa trên các tham số được chỉ định.
    """
    try:
        # Kiểm tra tính hợp lệ của cặp giao dịch và timeframe
        if not is_valid_trading_pair(symbol):
            logger.error(f"Cặp giao dịch không hợp lệ: {symbol}")
            sys.exit(1)
            
        if not is_valid_timeframe(timeframe):
            logger.error(f"Khung thời gian không hợp lệ: {timeframe}")
            sys.exit(1)
        
        # Thiết lập mức độ chi tiết
        verbose_level = int(verbose)
        
        # Thiết lập device cho các framework deep learning
        if gpu:
            try:
                import tensorflow as tf
                physical_devices = tf.config.list_physical_devices('GPU')
                if len(physical_devices) > 0:
                    tf.config.experimental.set_memory_growth(physical_devices[0], True)
                    logger.info(f"Sử dụng GPU: {physical_devices[0]}")
                else:
                    logger.warning("Không tìm thấy GPU, sử dụng CPU")
            except ImportError:
                logger.warning("TensorFlow không được cài đặt, sử dụng CPU")
        
        # Tải cấu hình huấn luyện từ file
        training_config = {}
        if config_file:
            with open(config_file, 'r', encoding='utf-8') as f:
                training_config = json.load(f)
                logger.info(f"Đã tải cấu hình từ {config_file}")
        
        # Tạo cấu hình mặc định và cập nhật từ tham số dòng lệnh
        default_config = {
            "num_episodes": episodes,
            "eval_frequency": eval_interval,
            "save_frequency": save_interval,
            "random_seed": random_seed,
            "render_eval": render,
            "render_frequency": 100 if render else 0,
        }
        
        # Cập nhật cấu hình mặc định với cấu hình từ file
        default_config.update(training_config)
        
        # Cấu hình cho agent
        agent_config = training_config.get("agent_config", {})
        agent_config.update({
            "random_seed": random_seed,
        })
        
        # Xác định đường dẫn dữ liệu
        if data_path is None:
            # Tìm dữ liệu mặc định dựa trên symbol và timeframe
            symbol_path = symbol.replace("/", "_")
            data_path = os.path.join(
                DEFAULT_DATA_DIR, 
                "historical", 
                symbol_path, 
                f"{symbol_path}_{timeframe}.csv"
            )
            if not os.path.exists(data_path):
                logger.error(f"Không tìm thấy file dữ liệu mặc định tại: {data_path}")
                logger.info("Vui lòng cung cấp đường dẫn dữ liệu bằng --data-path")
                sys.exit(1)
        
        # Tải dữ liệu
        logger.info(f"Đang tải dữ liệu từ {data_path}...")
        data = pd.read_csv(data_path)
        
        # Chuyển đổi cột thời gian nếu cần
        if 'timestamp' in data.columns and isinstance(data['timestamp'].iloc[0], str):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Tạo tên thí nghiệm nếu không được cung cấp
        if experiment_name is None:
            experiment_name = f"{agent_type}_{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Khởi tạo môi trường huấn luyện
        logger.info(f"Khởi tạo môi trường giao dịch với {len(data)} dòng dữ liệu...")
        env = TradingEnv(
            data=data,
            symbol=symbol,
            timeframe=timeframe,
            initial_balance=initial_balance,
            leverage=leverage,
            window_size=window_size,
            render_mode='human' if render else 'console'
        )
        
        # Động import agent class dựa trên agent_type
        if agent_type == AgentType.DQN.value:
            from models.agents.dqn_agent import DQNAgent
            agent_class = DQNAgent
        elif agent_type == AgentType.PPO.value:
            from models.agents.ppo_agent import PPOAgent
            agent_class = PPOAgent
        elif agent_type == AgentType.A2C.value:
            from models.agents.a2c_agent import A2CAgent
            agent_class = A2CAgent
        elif agent_type == AgentType.DDPG.value:
            from models.agents.ddpg_agent import DDPGAgent
            agent_class = DDPGAgent
        elif agent_type == AgentType.SAC.value:
            from models.agents.sac_agent import SACAgent
            agent_class = SACAgent
        elif agent_type == AgentType.TD3.value:
            from models.agents.td3_agent import TD3Agent
            agent_class = TD3Agent
        else:
            logger.error(f"Loại agent không được hỗ trợ: {agent_type}")
            sys.exit(1)
        
        # Khởi tạo agent
        logger.info(f"Khởi tạo agent {agent_type}...")
        
        # Chuẩn bị tham số cho agent
        agent_params = {
            "state_dim": env.observation_space.shape,
            "action_dim": env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0],
            "name": f"{agent_type}_{symbol.replace('/', '_')}",
        }
        agent_params.update(agent_config)
        
        # Tạo agent
        agent = agent_class(**agent_params)
        
        # Khởi tạo trainer
        logger.info(f"Khởi tạo trainer với {episodes} episodes...")
        trainer = Trainer(
            agent=agent,
            env=env,
            config=default_config,
            output_dir=output_dir,
            experiment_name=experiment_name,
            use_tensorboard=tensorboard,
            early_stopping=early_stopping,
            verbose=verbose_level
        )
        
        # Bắt đầu huấn luyện
        logger.info(f"Bắt đầu huấn luyện agent {agent_type} trên {symbol} ({timeframe})...")
        history = trainer.train()
        
        # Xuất báo cáo
        report_path = trainer.export_report()
        logger.info(f"Đã xuất báo cáo huấn luyện tại: {report_path}")
        
        # Lưu mô hình cuối cùng
        model_path = trainer.save_agent(suffix="final")
        logger.info(f"Đã lưu mô hình cuối cùng tại: {model_path}")
        
        # Tạo biểu đồ kết quả
        trainer.plot_training_results()
        logger.info(f"Đã tạo biểu đồ kết quả huấn luyện")
        
        # Hiển thị thông tin tóm tắt
        click.echo("\n" + "="*50)
        click.echo(f"Tóm tắt huấn luyện {agent_type}:")
        click.echo(f"  - Symbol: {symbol}")
        click.echo(f"  - Timeframe: {timeframe}")
        click.echo(f"  - Episodes: {trainer.current_episode}/{episodes}")
        if hasattr(history, "eval_rewards") and history["eval_rewards"]:
            click.echo(f"  - Phần thưởng đánh giá cuối cùng: {history['eval_rewards'][-1]:.2f}")
        click.echo(f"  - Mô hình tốt nhất tại episode: {trainer.best_episode}")
        click.echo(f"  - Báo cáo: {report_path}")
        click.echo("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện agent: {str(e)}", exc_info=True)
        sys.exit(1)

@train_commands.command(name="continue")
@click.argument("checkpoint_path", type=click.Path(exists=True))
@click.option("--episodes", "-e", type=int, default=500, help="Số lượng episodes bổ sung")
@click.option("--output-dir", "-o", type=click.Path(), help="Thư mục đầu ra mới (mặc định: sử dụng thư mục cũ)")
@click.option("--render/--no-render", default=False, help="Có hiển thị quá trình huấn luyện không")
@click.option("--gpu/--cpu", default=False, help="Sử dụng GPU hay không")
@click.option("--verbose", type=click.Choice(["0", "1", "2"]), default="1", help="Mức độ chi tiết (0: im lặng, 1: bình thường, 2: chi tiết)")
def continue_training(
    checkpoint_path: str,
    episodes: int,
    output_dir: Optional[str],
    render: bool,
    gpu: bool,
    verbose: str
):
    """
    Tiếp tục huấn luyện từ checkpoint.
    
    Lệnh này tiếp tục huấn luyện agent từ checkpoint đã lưu trước đó.
    """
    try:
        # Thiết lập mức độ chi tiết
        verbose_level = int(verbose)
        
        # Thiết lập device cho các framework deep learning
        if gpu:
            try:
                import tensorflow as tf
                physical_devices = tf.config.list_physical_devices('GPU')
                if len(physical_devices) > 0:
                    tf.config.experimental.set_memory_growth(physical_devices[0], True)
                    logger.info(f"Sử dụng GPU: {physical_devices[0]}")
                else:
                    logger.warning("Không tìm thấy GPU, sử dụng CPU")
            except ImportError:
                logger.warning("TensorFlow không được cài đặt, sử dụng CPU")
        
        # Tải thông tin checkpoint
        checkpoint_path = Path(checkpoint_path)
        
        # Xác định thư mục của checkpoint
        if 'checkpoints' in str(checkpoint_path):
            experiment_dir = checkpoint_path.parent.parent
        else:
            experiment_dir = checkpoint_path.parent
        
        # Tải cấu hình của thí nghiệm
        config_path = experiment_dir / "training_config.json"
        if not config_path.exists():
            logger.error(f"Không tìm thấy file cấu hình tại: {config_path}")
            sys.exit(1)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            training_config = json.load(f)
        
        # Tải lịch sử huấn luyện
        history_path = experiment_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r', encoding='utf-8') as f:
                training_history = json.load(f)
        else:
            training_history = {}
        
        # Lấy thông tin cơ bản về thí nghiệm
        agent_name = training_config.get("agent_name", "unknown_agent")
        env_name = training_config.get("env_name", "unknown_env")
        experiment_name = training_config.get("experiment_name", "unknown_experiment")
        
        # Xác định thư mục đầu ra
        if output_dir is None:
            new_output_dir = experiment_dir
        else:
            new_output_dir = Path(output_dir) / f"{experiment_name}_continued"
            new_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Trích xuất cấu hình của môi trường từ file cấu hình
        env_config = training_config.get("env_config", {})
        
        # Cập nhật cấu hình với tham số mới
        updated_config = training_config.get("training_config", {}).copy()
        updated_config["num_episodes"] = episodes
        updated_config["render_eval"] = render
        updated_config["render_frequency"] = 100 if render else 0
        
        logger.info(f"Tải lại môi trường: {env_name}")
        
        # Tạo lại môi trường
        if env_name == "TradingEnv":
            # Tải dữ liệu
            data_path = env_config.get("data_path")
            if data_path and os.path.exists(data_path):
                data = pd.read_csv(data_path)
            else:
                # Tìm dữ liệu từ thông tin trong cấu hình
                symbol = env_config.get("symbol", "BTC/USDT")
                timeframe = env_config.get("timeframe", "1h")
                symbol_path = symbol.replace("/", "_")
                default_data_path = os.path.join(
                    DEFAULT_DATA_DIR, 
                    "historical", 
                    symbol_path, 
                    f"{symbol_path}_{timeframe}.csv"
                )
                
                if os.path.exists(default_data_path):
                    data = pd.read_csv(default_data_path)
                else:
                    logger.error(f"Không tìm thấy dữ liệu tại: {default_data_path}")
                    sys.exit(1)
            
            # Chuyển đổi cột thời gian nếu cần
            if 'timestamp' in data.columns and isinstance(data['timestamp'].iloc[0], str):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Khởi tạo lại môi trường
            env = TradingEnv(
                data=data,
                symbol=env_config.get("symbol", "BTC/USDT"),
                timeframe=env_config.get("timeframe", "1h"),
                initial_balance=env_config.get("initial_balance", 10000.0),
                leverage=env_config.get("leverage", 1.0),
                window_size=env_config.get("window_size", 100),
                render_mode='human' if render else 'console'
            )
        else:
            logger.error(f"Không hỗ trợ môi trường: {env_name}")
            sys.exit(1)
        
        # Tìm và import lớp agent
        agent_class_name = agent_name.split('_')[0] if '_' in agent_name else agent_name
        
        # Map tên agent sang module path
        agent_module_map = {
            "DQNAgent": "models.agents.dqn_agent",
            "PPOAgent": "models.agents.ppo_agent",
            "A2CAgent": "models.agents.a2c_agent",
            "DDPGAgent": "models.agents.ddpg_agent",
            "SACAgent": "models.agents.sac_agent",
            "TD3Agent": "models.agents.td3_agent",
        }
        
        # Tìm module phù hợp
        agent_module_path = None
        for class_name, module_path in agent_module_map.items():
            if class_name.lower() in agent_name.lower():
                agent_module_path = module_path
                agent_class_name = class_name
                break
        
        if agent_module_path is None:
            logger.error(f"Không tìm thấy module phù hợp cho agent: {agent_name}")
            sys.exit(1)
        
        # Import agent class
        agent_module = importlib.import_module(agent_module_path)
        agent_class = getattr(agent_module, agent_class_name)
        
        # Khởi tạo agent
        agent_config = training_config.get("agent_config", {})
        agent_params = {
            "state_dim": env.observation_space.shape,
            "action_dim": env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0],
            "name": agent_name,
        }
        agent_params.update(agent_config)
        
        logger.info(f"Khởi tạo agent: {agent_class_name}")
        agent = agent_class(**agent_params)
        
        # Khởi tạo trainer
        logger.info(f"Khởi tạo trainer với cấu hình tiếp tục huấn luyện...")
        trainer = Trainer(
            agent=agent,
            env=env,
            config=updated_config,
            output_dir=new_output_dir,
            experiment_name=f"{experiment_name}_continued",
            verbose=verbose_level
        )
        
        # Tải lại checkpoint
        logger.info(f"Tải checkpoint từ: {checkpoint_path}")
        success = trainer.restore_checkpoint(checkpoint_path)
        
        if not success:
            logger.error(f"Không thể tải checkpoint từ: {checkpoint_path}")
            sys.exit(1)
        
        # Bắt đầu huấn luyện tiếp tục
        logger.info(f"Tiếp tục huấn luyện với {episodes} episodes bổ sung...")
        history = trainer.train()
        
        # Xuất báo cáo
        report_path = trainer.export_report()
        logger.info(f"Đã xuất báo cáo huấn luyện tại: {report_path}")
        
        # Lưu mô hình cuối cùng
        model_path = trainer.save_agent(suffix="continued_final")
        logger.info(f"Đã lưu mô hình cuối cùng tại: {model_path}")
        
        # Tạo biểu đồ kết quả
        trainer.plot_training_results()
        logger.info(f"Đã tạo biểu đồ kết quả huấn luyện")
        
        # Hiển thị thông tin tóm tắt
        click.echo("\n" + "="*50)
        click.echo(f"Tóm tắt tiếp tục huấn luyện {agent_name}:")
        click.echo(f"  - Episodes bổ sung: {episodes}")
        click.echo(f"  - Tổng số episodes: {trainer.current_episode}")
        if hasattr(history, "eval_rewards") and history["eval_rewards"]:
            click.echo(f"  - Phần thưởng đánh giá cuối cùng: {history['eval_rewards'][-1]:.2f}")
        click.echo(f"  - Mô hình tốt nhất tại episode: {trainer.best_episode}")
        click.echo(f"  - Báo cáo: {report_path}")
        click.echo("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Lỗi khi tiếp tục huấn luyện: {str(e)}", exc_info=True)
        sys.exit(1)

@train_commands.command(name="optimize")
@click.argument("agent_type", type=click.Choice([agent.value for agent in AgentType]))
@click.option("--data-path", "-d", type=click.Path(exists=True), help="Đường dẫn đến file dữ liệu huấn luyện")
@click.option("--symbol", "-s", type=str, default="BTC/USDT", help="Cặp giao dịch (ví dụ: BTC/USDT)")
@click.option("--timeframe", "-t", type=str, default="1h", help="Khung thời gian (1m, 5m, 15m, 1h, 4h, 1d, ...)")
@click.option("--output-dir", "-o", type=click.Path(), default=str(DEFAULT_OUTPUT_DIR), help="Thư mục đầu ra cho kết quả tối ưu hóa")
@click.option("--config-file", "-c", type=click.Path(exists=True), help="File cấu hình cho tối ưu hóa (JSON)")
@click.option("--method", "-m", type=click.Choice(["grid", "random", "bayesian", "optuna"]), default="optuna", help="Phương pháp tối ưu hóa")
@click.option("--trials", "-n", type=int, default=20, help="Số lượng thử nghiệm")
@click.option("--episodes", "-e", type=int, default=300, help="Số lượng episodes cho mỗi thử nghiệm")
@click.option("--n-jobs", "-j", type=int, default=1, help="Số lượng công việc chạy song song")
@click.option("--random-seed", "-r", type=int, default=42, help="Seed cho bộ sinh số ngẫu nhiên")
@click.option("--verbose", type=click.Choice(["0", "1", "2"]), default="1", help="Mức độ chi tiết (0: im lặng, 1: bình thường, 2: chi tiết)")
@click.option("--experiment-name", "-n", type=str, help="Tên thí nghiệm")
def optimize_hyperparameters(
    agent_type: str,
    data_path: Optional[str],
    symbol: str,
    timeframe: str,
    output_dir: str,
    config_file: Optional[str],
    method: str,
    trials: int,
    episodes: int,
    n_jobs: int,
    random_seed: int,
    verbose: str,
    experiment_name: Optional[str]
):
    """
    Tối ưu hóa siêu tham số cho agent.
    
    Lệnh này thực hiện tối ưu hóa siêu tham số cho agent dựa trên phương pháp được chỉ định.
    """
    try:
        # Thiết lập mức độ chi tiết
        verbose_level = int(verbose)
        
        # Kiểm tra tính hợp lệ của cặp giao dịch và timeframe
        if not is_valid_trading_pair(symbol):
            logger.error(f"Cặp giao dịch không hợp lệ: {symbol}")
            sys.exit(1)
            
        if not is_valid_timeframe(timeframe):
            logger.error(f"Khung thời gian không hợp lệ: {timeframe}")
            sys.exit(1)
        
        # Tạo tên thí nghiệm nếu không được cung cấp
        if experiment_name is None:
            experiment_name = f"optimize_{agent_type}_{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Xác định đường dẫn dữ liệu
        if data_path is None:
            # Tìm dữ liệu mặc định dựa trên symbol và timeframe
            symbol_path = symbol.replace("/", "_")
            data_path = os.path.join(
                DEFAULT_DATA_DIR, 
                "historical", 
                symbol_path, 
                f"{symbol_path}_{timeframe}.csv"
            )
            if not os.path.exists(data_path):
                logger.error(f"Không tìm thấy file dữ liệu mặc định tại: {data_path}")
                logger.info("Vui lòng cung cấp đường dẫn dữ liệu bằng --data-path")
                sys.exit(1)
        
        # Tải dữ liệu
        logger.info(f"Đang tải dữ liệu từ {data_path}...")
        data = pd.read_csv(data_path)
        
        # Chuyển đổi cột thời gian nếu cần
        if 'timestamp' in data.columns and isinstance(data['timestamp'].iloc[0], str):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Tìm và import lớp agent
        if agent_type == AgentType.DQN.value:
            from models.agents.dqn_agent import DQNAgent
            agent_class = DQNAgent
        elif agent_type == AgentType.PPO.value:
            from models.agents.ppo_agent import PPOAgent
            agent_class = PPOAgent
        elif agent_type == AgentType.A2C.value:
            from models.agents.a2c_agent import A2CAgent
            agent_class = A2CAgent
        elif agent_type == AgentType.DDPG.value:
            from models.agents.ddpg_agent import DDPGAgent
            agent_class = DDPGAgent
        elif agent_type == AgentType.SAC.value:
            from models.agents.sac_agent import SACAgent
            agent_class = SACAgent
        elif agent_type == AgentType.TD3.value:
            from models.agents.td3_agent import TD3Agent
            agent_class = TD3Agent
        else:
            logger.error(f"Loại agent không được hỗ trợ: {agent_type}")
            sys.exit(1)
        
        # Lớp môi trường
        env_class = TradingEnv
        
        # Tải cấu hình tối ưu hóa từ file
        param_space = {}
        if config_file:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                param_space = config_data.get("param_space", {})
                logger.info(f"Đã tải không gian tham số từ {config_file}")
        
        # Sử dụng không gian tham số mặc định nếu không tìm thấy trong file
        if not param_space:
            # Không gian tham số mặc định cho từng loại agent
            if agent_type == AgentType.DQN.value:
                param_space = {
                    "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-2, "log_scale": True},
                    "gamma": {"type": "float", "min": 0.9, "max": 0.9999},
                    "epsilon_start": {"type": "float", "min": 0.8, "max": 1.0},
                    "epsilon_end": {"type": "float", "min": 0.01, "max": 0.2},
                    "epsilon_decay": {"type": "int", "min": 1000, "max": 100000, "step": 1000},
                    "batch_size": {"type": "int", "min": 16, "max": 512, "step": 16},
                    "memory_size": {"type": "int", "min": 1000, "max": 50000, "step": 1000},
                    "hidden_size": {"type": "int", "min": 32, "max": 512, "step": 32},
                    "target_update": {"type": "int", "min": 10, "max": 1000, "step": 10},
                }
            elif agent_type == AgentType.PPO.value:
                param_space = {
                    "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-3, "log_scale": True},
                    "gamma": {"type": "float", "min": 0.9, "max": 0.999},
                    "clip_ratio": {"type": "float", "min": 0.1, "max": 0.3},
                    "value_coef": {"type": "float", "min": 0.3, "max": 0.8},
                    "entropy_coef": {"type": "float", "min": 0.0, "max": 0.1},
                    "batch_size": {"type": "int", "min": 64, "max": 1024, "step": 64},
                    "hidden_size": {"type": "int", "min": 32, "max": 512, "step": 32},
                    "policy_update_iterations": {"type": "int", "min": 3, "max": 10},
                }
            elif agent_type == AgentType.A2C.value:
                param_space = {
                    "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-3, "log_scale": True},
                    "gamma": {"type": "float", "min": 0.9, "max": 0.999},
                    "value_coef": {"type": "float", "min": 0.3, "max": 0.8},
                    "entropy_coef": {"type": "float", "min": 0.0, "max": 0.1},
                    "hidden_size": {"type": "int", "min": 32, "max": 512, "step": 32},
                    "update_steps": {"type": "int", "min": 5, "max": 20},
                }
            else:
                # Không gian tham số mặc định chung cho các agent khác
                param_space = {
                    "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-2, "log_scale": True},
                    "gamma": {"type": "float", "min": 0.9, "max": 0.999},
                    "batch_size": {"type": "int", "min": 16, "max": 256, "step": 16},
                    "hidden_size": {"type": "int", "min": 32, "max": 512, "step": 32},
                }
            
            logger.info(f"Sử dụng không gian tham số mặc định cho {agent_type}")
        
        # Cấu hình cho tối ưu hóa
        env_kwargs = {
            "data": data,
            "symbol": symbol,
            "timeframe": timeframe,
            "initial_balance": 10000.0,
            "leverage": 1.0,
            "window_size": 100,
            "render_mode": 'console'
        }
        
        # Cấu hình mặc định cho trainer
        trainer_kwargs = {
            "num_episodes": episodes,
            "eval_frequency": 10,
            "save_frequency": episodes // 2,
            "use_tensorboard": False,
            "verbose": verbose_level,
            "early_stopping": True,
            "patience": 5
        }
        
        # Khởi tạo HyperparameterTuner
        logger.info(f"Khởi tạo HyperparameterTuner với phương pháp {method}...")
        tuner = HyperparameterTuner(
            agent_class=agent_class,
            env_class=env_class,
            param_space=param_space,
            metric="eval_reward_mean",
            metric_direction="maximize",
            n_trials=trials,
            random_seed=random_seed,
            output_dir=output_dir,
            n_jobs=n_jobs,
            logger=logger,
            experiment_name=experiment_name,
            env_kwargs=env_kwargs,
            trainer_kwargs=trainer_kwargs
        )
        
        # Thực hiện tối ưu hóa
        logger.info(f"Bắt đầu tối ưu hóa với {trials} thử nghiệm...")
        
        if method == "grid":
            result = tuner.grid_search()
        elif method == "random":
            result = tuner.random_search()
        elif method == "bayesian":
            result = tuner.bayesian_optimization()
        elif method == "optuna":
            result = tuner.optuna_optimization()
        
        # Xuất báo cáo
        report_path = tuner.export_report()
        logger.info(f"Đã xuất báo cáo tối ưu hóa tại: {report_path}")
        
        # Huấn luyện agent với tham số tốt nhất
        logger.info(f"Huấn luyện agent cuối cùng với tham số tốt nhất...")
        best_trainer = tuner.train_best_agent(num_episodes=episodes * 2)
        
        # Hiển thị thông tin tóm tắt
        click.echo("\n" + "="*50)
        click.echo(f"Tóm tắt tối ưu hóa siêu tham số cho {agent_type}:")
        click.echo(f"  - Symbol: {symbol}")
        click.echo(f"  - Timeframe: {timeframe}")
        click.echo(f"  - Phương pháp: {method}")
        click.echo(f"  - Số thử nghiệm: {trials}")
        click.echo(f"  - Tham số tốt nhất:")
        
        for param, value in result["best_params"].items():
            click.echo(f"    - {param}: {value}")
        
        click.echo(f"  - Phần thưởng tốt nhất: {result['best_score']:.4f}")
        click.echo(f"  - Báo cáo: {report_path}")
        click.echo("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Lỗi khi tối ưu hóa siêu tham số: {str(e)}", exc_info=True)
        sys.exit(1)

@train_commands.command(name="evaluate")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--data-path", "-d", type=click.Path(exists=True), help="Đường dẫn đến file dữ liệu đánh giá")
@click.option("--output-dir", "-o", type=click.Path(), help="Thư mục đầu ra cho kết quả đánh giá")
@click.option("--episodes", "-e", type=int, default=10, help="Số lượng episodes đánh giá")
@click.option("--render/--no-render", default=True, help="Có hiển thị quá trình đánh giá không")
@click.option("--verbose", type=click.Choice(["0", "1", "2"]), default="1", help="Mức độ chi tiết (0: im lặng, 1: bình thường, 2: chi tiết)")
def evaluate_model(
    model_path: str,
    data_path: Optional[str],
    output_dir: Optional[str],
    episodes: int,
    render: bool,
    verbose: str
):
    """
    Đánh giá mô hình đã huấn luyện.
    
    Lệnh này thực hiện đánh giá mô hình trên tập dữ liệu mới hoặc hiện có.
    """
    try:
        # Thiết lập mức độ chi tiết
        verbose_level = int(verbose)
        
        # Tải thông tin mô hình
        model_path = Path(model_path)
        
        # Xác định thư mục của mô hình
        if model_path.is_file():
            model_dir = model_path.parent
            if model_dir.name == 'models':
                experiment_dir = model_dir.parent
            else:
                experiment_dir = model_dir
        else:
            experiment_dir = model_path
            model_path = list(model_path.glob("best_model*"))[0] if list(model_path.glob("best_model*")) else None
            
            if model_path is None:
                logger.error(f"Không tìm thấy mô hình trong thư mục: {experiment_dir}")
                sys.exit(1)
        
        # Tải cấu hình của thí nghiệm
        config_path = experiment_dir / "training_config.json"
        if not config_path.exists():
            logger.error(f"Không tìm thấy file cấu hình tại: {config_path}")
            sys.exit(1)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            training_config = json.load(f)
        
        # Lấy thông tin cơ bản về thí nghiệm
        agent_name = training_config.get("agent_name", "unknown_agent")
        env_name = training_config.get("env_name", "unknown_env")
        experiment_name = training_config.get("experiment_name", "unknown_experiment")
        
        # Xác định thư mục đầu ra
        if output_dir is None:
            output_dir = experiment_dir / "evaluation"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Trích xuất cấu hình của môi trường từ file cấu hình
        env_config = training_config.get("env_config", {})
        
        # Chuẩn bị dữ liệu đánh giá
        if data_path is None:
            # Sử dụng lại dữ liệu huấn luyện nếu không có dữ liệu đánh giá mới
            data_path = env_config.get("data_path")
            
            if not data_path or not os.path.exists(data_path):
                # Tìm dữ liệu từ thông tin trong cấu hình
                symbol = env_config.get("symbol", "BTC/USDT")
                timeframe = env_config.get("timeframe", "1h")
                symbol_path = symbol.replace("/", "_")
                default_data_path = os.path.join(
                    DEFAULT_DATA_DIR, 
                    "historical", 
                    symbol_path, 
                    f"{symbol_path}_{timeframe}.csv"
                )
                
                if os.path.exists(default_data_path):
                    data_path = default_data_path
                else:
                    logger.error(f"Không tìm thấy dữ liệu huấn luyện gốc và không cung cấp dữ liệu đánh giá")
                    sys.exit(1)
        
        # Tải dữ liệu
        logger.info(f"Đang tải dữ liệu từ {data_path}...")
        data = pd.read_csv(data_path)
        
        # Chuyển đổi cột thời gian nếu cần
        if 'timestamp' in data.columns and isinstance(data['timestamp'].iloc[0], str):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Tạo lại môi trường
        if env_name == "TradingEnv":
            # Lấy thông tin về symbol, timeframe từ cấu hình
            symbol = env_config.get("symbol", "BTC/USDT")
            timeframe = env_config.get("timeframe", "1h")
            initial_balance = env_config.get("initial_balance", 10000.0)
            leverage = env_config.get("leverage", 1.0)
            window_size = env_config.get("window_size", 100)
            
            # Khởi tạo lại môi trường
            env = TradingEnv(
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                initial_balance=initial_balance,
                leverage=leverage,
                window_size=window_size,
                render_mode='human' if render else 'console'
            )
        else:
            logger.error(f"Không hỗ trợ môi trường: {env_name}")
            sys.exit(1)
        
        # Tìm và import lớp agent
        agent_class_name = agent_name.split('_')[0] if '_' in agent_name else agent_name
        
        # Map tên agent sang module path
        agent_module_map = {
            "DQNAgent": "models.agents.dqn_agent",
            "PPOAgent": "models.agents.ppo_agent",
            "A2CAgent": "models.agents.a2c_agent",
            "DDPGAgent": "models.agents.ddpg_agent",
            "SACAgent": "models.agents.sac_agent",
            "TD3Agent": "models.agents.td3_agent",
        }
        
        # Tìm module phù hợp
        agent_module_path = None
        for class_name, module_path in agent_module_map.items():
            if class_name.lower() in agent_name.lower():
                agent_module_path = module_path
                agent_class_name = class_name
                break
        
        if agent_module_path is None:
            logger.error(f"Không tìm thấy module phù hợp cho agent: {agent_name}")
            sys.exit(1)
        
        # Import agent class
        agent_module = importlib.import_module(agent_module_path)
        agent_class = getattr(agent_module, agent_class_name)
        
        # Khởi tạo agent
        agent_config = training_config.get("agent_config", {})
        agent_params = {
            "state_dim": env.observation_space.shape,
            "action_dim": env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0],
            "name": agent_name,
        }
        agent_params.update(agent_config)
        
        logger.info(f"Khởi tạo agent: {agent_class_name}")
        agent = agent_class(**agent_params)
        
        # Tải mô hình
        logger.info(f"Tải mô hình từ: {model_path}")
        success = agent._load_model_impl(model_path)
        
        if not success:
            logger.error(f"Không thể tải mô hình từ: {model_path}")
            sys.exit(1)
        
        # Tạo thư mục kết quả
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Thực hiện đánh giá
        logger.info(f"Bắt đầu đánh giá với {episodes} episodes...")
        
        # Danh sách lưu kết quả
        episode_rewards = []
        episode_returns = []
        episode_lengths = []
        win_trades = 0
        loss_trades = 0
        total_trades = 0
        
        # Thực hiện đánh giá
        for episode in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            
            while not done:
                # Lấy hành động từ agent
                action = agent.act(state, explore=False)
                
                # Thực thi hành động
                next_state, reward, done, info = env.step(action)
                
                # Cập nhật trạng thái và thông tin
                state = next_state
                episode_reward += reward
                episode_step += 1
                
                # Hiển thị quá trình đánh giá
                if render:
                    env.render()
            
            # Tính toán lợi nhuận và các chỉ số khác
            final_balance = env.current_balance
            returns = (final_balance / env.initial_balance - 1) * 100  # % lợi nhuận
            
            # Lưu thống kê giao dịch
            if hasattr(env, 'order_history') and env.order_history:
                for trade in env.order_history:
                    if trade['pnl'] > 0:
                        win_trades += 1
                    else:
                        loss_trades += 1
                    total_trades += 1
            
            # Lưu kết quả
            episode_rewards.append(episode_reward)
            episode_returns.append(returns)
            episode_lengths.append(episode_step)
            
            # Hiển thị kết quả episode
            logger.info(f"Episode {episode+1}/{episodes} - Reward: {episode_reward:.2f}, Return: {returns:.2f}%, Steps: {episode_step}")
        
        # Tính toán thống kê
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        mean_length = np.mean(episode_lengths)
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        # Lưu kết quả vào file
        results = {
            "model_path": str(model_path),
            "data_path": data_path,
            "episodes": episodes,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "mean_return": float(mean_return),
            "std_return": float(std_return),
            "mean_length": float(mean_length),
            "win_rate": float(win_rate),
            "total_trades": total_trades,
            "win_trades": win_trades,
            "loss_trades": loss_trades,
            "episode_rewards": episode_rewards,
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
            "timestamp": datetime.now().isoformat()
        }
        
        # Lưu kết quả
        results_file = results_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, default=lambda x: float(x) if isinstance(x, np.number) else x)
        
        logger.info(f"Đã lưu kết quả đánh giá tại: {results_file}")
        
        # Tạo biểu đồ kết quả
        plt.figure(figsize=(15, 10))
        
        # Biểu đồ phần thưởng
        plt.subplot(2, 2, 1)
        plt.plot(episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        
        # Biểu đồ lợi nhuận
        plt.subplot(2, 2, 2)
        plt.plot(episode_returns)
        plt.title('Episode Returns (%)')
        plt.xlabel('Episode')
        plt.ylabel('Return (%)')
        plt.grid(True, alpha=0.3)
        
        # Biểu đồ độ dài episode
        plt.subplot(2, 2, 3)
        plt.plot(episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True, alpha=0.3)
        
        # Biểu đồ tỷ lệ thắng/thua
        plt.subplot(2, 2, 4)
        if total_trades > 0:
            plt.pie([win_trades, loss_trades], labels=['Win', 'Loss'], autopct='%1.1f%%')
            plt.title('Win/Loss Ratio')
        else:
            plt.text(0.5, 0.5, 'No trades', horizontalalignment='center', verticalalignment='center')
            plt.title('Win/Loss Ratio (No Trades)')
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        chart_path = results_dir / f"evaluation_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path)
        logger.info(f"Đã lưu biểu đồ kết quả tại: {chart_path}")
        
        # Hiển thị thông tin tóm tắt
        click.echo("\n" + "="*50)
        click.echo(f"Tóm tắt đánh giá {agent_name}:")
        click.echo(f"  - Model: {model_path}")
        click.echo(f"  - Số lượng episodes: {episodes}")
        click.echo(f"  - Phần thưởng trung bình: {mean_reward:.2f} ± {std_reward:.2f}")
        click.echo(f"  - Lợi nhuận trung bình: {mean_return:.2f}% ± {std_return:.2f}%")
        click.echo(f"  - Số giao dịch: {total_trades}")
        click.echo(f"  - Tỷ lệ thắng: {win_rate:.2%} ({win_trades}/{total_trades})")
        click.echo(f"  - Kết quả đánh giá: {results_file}")
        click.echo(f"  - Biểu đồ: {chart_path}")
        click.echo("="*50 + "\n")
        
        # Đóng môi trường
        env.close()
        
    except Exception as e:
        logger.error(f"Lỗi khi đánh giá mô hình: {str(e)}", exc_info=True)
        sys.exit(1)

@train_commands.command(name="compare")
@click.argument("model_paths", type=click.Path(exists=True), nargs=-1)
@click.option("--data-path", "-d", type=click.Path(exists=True), help="Đường dẫn đến file dữ liệu đánh giá")
@click.option("--output-dir", "-o", type=click.Path(), help="Thư mục đầu ra cho kết quả so sánh")
@click.option("--episodes", "-e", type=int, default=10, help="Số lượng episodes đánh giá cho mỗi mô hình")
@click.option("--render/--no-render", default=False, help="Có hiển thị quá trình đánh giá không")
@click.option("--verbose", type=click.Choice(["0", "1", "2"]), default="1", help="Mức độ chi tiết (0: im lặng, 1: bình thường, 2: chi tiết)")
def compare_models(
    model_paths: List[str],
    data_path: Optional[str],
    output_dir: Optional[str],
    episodes: int,
    render: bool,
    verbose: str
):
    """
    So sánh nhiều mô hình đã huấn luyện.
    
    Lệnh này thực hiện đánh giá và so sánh nhiều mô hình trên cùng một tập dữ liệu.
    """
    try:
        # Kiểm tra số lượng mô hình
        if len(model_paths) < 2:
            logger.error("Cần ít nhất 2 mô hình để so sánh")
            sys.exit(1)
        
        # Thiết lập mức độ chi tiết
        verbose_level = int(verbose)
        
        # Xác định thư mục đầu ra
        if output_dir is None:
            output_dir = Path(f"./model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Các biến lưu trữ kết quả
        model_results = {}
        model_configs = {}
        env_configs = {}
        
        # Xác định dữ liệu đánh giá chung
        if data_path is None:
            # Tìm dữ liệu mặc định từ mô hình đầu tiên
            first_model_path = Path(model_paths[0])
            
            if first_model_path.is_file():
                experiment_dir = first_model_path.parent.parent
            else:
                experiment_dir = first_model_path
            
            config_path = experiment_dir / "training_config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    first_config = json.load(f)
                
                env_config = first_config.get("env_config", {})
                default_data_path = env_config.get("data_path")
                
                if default_data_path and os.path.exists(default_data_path):
                    data_path = default_data_path
                else:
                    # Tìm dữ liệu từ thông tin trong cấu hình
                    symbol = env_config.get("symbol", "BTC/USDT")
                    timeframe = env_config.get("timeframe", "1h")
                    symbol_path = symbol.replace("/", "_")
                    default_data_path = os.path.join(
                        DEFAULT_DATA_DIR, 
                        "historical", 
                        symbol_path, 
                        f"{symbol_path}_{timeframe}.csv"
                    )
                    
                    if os.path.exists(default_data_path):
                        data_path = default_data_path
            
            if data_path is None:
                logger.error("Không thể tìm thấy dữ liệu đánh giá mặc định, vui lòng cung cấp đường dẫn dữ liệu")
                sys.exit(1)
        
        # Tải dữ liệu
        logger.info(f"Đang tải dữ liệu từ {data_path}...")
        data = pd.read_csv(data_path)
        
        # Chuyển đổi cột thời gian nếu cần
        if 'timestamp' in data.columns and isinstance(data['timestamp'].iloc[0], str):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Thực hiện đánh giá cho từng mô hình
        for i, model_path in enumerate(model_paths):
            model_path = Path(model_path)
            model_name = f"Model_{i+1}"
            
            logger.info(f"Đánh giá {model_name}: {model_path}")
            
            try:
                # Xác định thư mục của mô hình
                if model_path.is_file():
                    model_dir = model_path.parent
                    if model_dir.name == 'models':
                        experiment_dir = model_dir.parent
                    else:
                        experiment_dir = model_dir
                else:
                    experiment_dir = model_path
                    model_path = list(model_path.glob("best_model*"))[0] if list(model_path.glob("best_model*")) else None
                    
                    if model_path is None:
                        logger.error(f"Không tìm thấy mô hình trong thư mục: {experiment_dir}")
                        continue
                
                # Tải cấu hình của thí nghiệm
                config_path = experiment_dir / "training_config.json"
                if not config_path.exists():
                    logger.error(f"Không tìm thấy file cấu hình tại: {config_path}")
                    continue
                
                with open(config_path, 'r', encoding='utf-8') as f:
                    training_config = json.load(f)
                
                # Lưu cấu hình
                model_configs[model_name] = training_config
                
                # Lấy thông tin cơ bản về thí nghiệm
                agent_name = training_config.get("agent_name", "unknown_agent")
                env_name = training_config.get("env_name", "unknown_env")
                experiment_name = training_config.get("experiment_name", "unknown_experiment")
                
                # Cấu hình môi trường từ file cấu hình
                env_config = training_config.get("env_config", {})
                env_configs[model_name] = env_config
                
                # Tạo lại môi trường
                if env_name == "TradingEnv":
                    # Lấy thông tin về symbol, timeframe từ cấu hình
                    symbol = env_config.get("symbol", "BTC/USDT")
                    timeframe = env_config.get("timeframe", "1h")
                    initial_balance = env_config.get("initial_balance", 10000.0)
                    leverage = env_config.get("leverage", 1.0)
                    window_size = env_config.get("window_size", 100)
                    
                    # Khởi tạo lại môi trường
                    env = TradingEnv(
                        data=data,
                        symbol=symbol,
                        timeframe=timeframe,
                        initial_balance=initial_balance,
                        leverage=leverage,
                        window_size=window_size,
                        render_mode='human' if render else 'console'
                    )
                    
                    # Cập nhật tên hiển thị
                    if "experiment_name" in training_config:
                        model_name = f"{training_config['experiment_name']}"
                else:
                    logger.error(f"Không hỗ trợ môi trường: {env_name}")
                    continue
                
                # Tìm và import lớp agent
                agent_class_name = agent_name.split('_')[0] if '_' in agent_name else agent_name
                
                # Map tên agent sang module path
                agent_module_map = {
                    "DQNAgent": "models.agents.dqn_agent",
                    "PPOAgent": "models.agents.ppo_agent",
                    "A2CAgent": "models.agents.a2c_agent",
                    "DDPGAgent": "models.agents.ddpg_agent",
                    "SACAgent": "models.agents.sac_agent",
                    "TD3Agent": "models.agents.td3_agent",
                }
                
                # Tìm module phù hợp
                agent_module_path = None
                for class_name, module_path in agent_module_map.items():
                    if class_name.lower() in agent_name.lower():
                        agent_module_path = module_path
                        agent_class_name = class_name
                        break
                
                if agent_module_path is None:
                    logger.error(f"Không tìm thấy module phù hợp cho agent: {agent_name}")
                    continue
                
                # Import agent class
                agent_module = importlib.import_module(agent_module_path)
                agent_class = getattr(agent_module, agent_class_name)
                
                # Khởi tạo agent
                agent_config = training_config.get("agent_config", {})
                agent_params = {
                    "state_dim": env.observation_space.shape,
                    "action_dim": env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0],
                    "name": agent_name,
                }
                agent_params.update(agent_config)
                
                logger.info(f"Khởi tạo agent: {agent_class_name}")
                agent = agent_class(**agent_params)
                
                # Tải mô hình
                logger.info(f"Tải mô hình từ: {model_path}")
                success = agent._load_model_impl(model_path)
                
                if not success:
                    logger.error(f"Không thể tải mô hình từ: {model_path}")
                    continue
                
                # Danh sách lưu kết quả
                episode_rewards = []
                episode_returns = []
                episode_lengths = []
                win_trades = 0
                loss_trades = 0
                total_trades = 0
                trade_pnls = []
                
                # Thực hiện đánh giá
                for episode in range(episodes):
                    state = env.reset()
                    done = False
                    episode_reward = 0
                    episode_step = 0
                    
                    while not done:
                        # Lấy hành động từ agent
                        action = agent.act(state, explore=False)
                        
                        # Thực thi hành động
                        next_state, reward, done, info = env.step(action)
                        
                        # Cập nhật trạng thái và thông tin
                        state = next_state
                        episode_reward += reward
                        episode_step += 1
                        
                        # Hiển thị quá trình đánh giá
                        if render:
                            env.render()
                    
                    # Tính toán lợi nhuận và các chỉ số khác
                    final_balance = env.current_balance
                    returns = (final_balance / env.initial_balance - 1) * 100  # % lợi nhuận
                    
                    # Lưu thống kê giao dịch
                    if hasattr(env, 'order_history') and env.order_history:
                        for trade in env.order_history:
                            trade_pnls.append(trade['pnl'])
                            if trade['pnl'] > 0:
                                win_trades += 1
                            else:
                                loss_trades += 1
                            total_trades += 1
                    
                    # Lưu kết quả
                    episode_rewards.append(episode_reward)
                    episode_returns.append(returns)
                    episode_lengths.append(episode_step)
                    
                    # Hiển thị kết quả episode
                    logger.info(f"{model_name} - Episode {episode+1}/{episodes} - Reward: {episode_reward:.2f}, Return: {returns:.2f}%, Steps: {episode_step}")
                
                # Tính toán thống kê
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                mean_return = np.mean(episode_returns)
                std_return = np.std(episode_returns)
                mean_length = np.mean(episode_lengths)
                win_rate = win_trades / total_trades if total_trades > 0 else 0
                
                # Tính các chỉ số phân tích bổ sung
                avg_pnl = np.mean(trade_pnls) if trade_pnls else 0
                max_pnl = np.max(trade_pnls) if trade_pnls else 0
                min_pnl = np.min(trade_pnls) if trade_pnls else 0
                
                # Tính drawdown
                if hasattr(env, 'performance_metrics') and 'max_drawdown' in env.performance_metrics:
                    max_drawdown = env.performance_metrics['max_drawdown']
                else:
                    max_drawdown = 0
                
                # Lưu kết quả
                model_results[model_name] = {
                    "model_path": str(model_path),
                    "mean_reward": float(mean_reward),
                    "std_reward": float(std_reward),
                    "mean_return": float(mean_return),
                    "std_return": float(std_return),
                    "mean_length": float(mean_length),
                    "win_rate": float(win_rate),
                    "total_trades": total_trades,
                    "win_trades": win_trades,
                    "loss_trades": loss_trades,
                    "episode_rewards": [float(r) for r in episode_rewards],
                    "episode_returns": [float(r) for r in episode_returns],
                    "episode_lengths": episode_lengths,
                    "avg_pnl": float(avg_pnl),
                    "max_pnl": float(max_pnl),
                    "min_pnl": float(min_pnl),
                    "max_drawdown": float(max_drawdown)
                }
                
                # Đóng môi trường
                env.close()
                
            except Exception as e:
                logger.error(f"Lỗi khi đánh giá {model_name}: {str(e)}", exc_info=True)
                continue
        
        # Kiểm tra xem có đủ mô hình để so sánh không
        if len(model_results) < 2:
            logger.error("Không đủ mô hình hợp lệ để so sánh")
            sys.exit(1)
        
        # Lưu kết quả so sánh
        comparison_results = {
            "data_path": data_path,
            "episodes": episodes,
            "model_results": model_results,
            "timestamp": datetime.now().isoformat()
        }
        
        results_file = output_dir / f"model_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=4, default=lambda x: float(x) if isinstance(x, np.number) else x)
        
        logger.info(f"Đã lưu kết quả so sánh tại: {results_file}")
        
        # Tạo biểu đồ so sánh
        plt.figure(figsize=(15, 10))
        
        # Danh sách màu
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Biểu đồ phần thưởng trung bình
        plt.subplot(2, 2, 1)
        model_names = list(model_results.keys())
        mean_rewards = [model_results[name]["mean_reward"] for name in model_names]
        std_rewards = [model_results[name]["std_reward"] for name in model_names]
        
        plt.bar(range(len(model_names)), mean_rewards, yerr=std_rewards, capsize=10, color=colors[:len(model_names)])
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.title('Mean Reward')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        
        # Biểu đồ lợi nhuận trung bình
        plt.subplot(2, 2, 2)
        mean_returns = [model_results[name]["mean_return"] for name in model_names]
        std_returns = [model_results[name]["std_return"] for name in model_names]
        
        plt.bar(range(len(model_names)), mean_returns, yerr=std_returns, capsize=10, color=colors[:len(model_names)])
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.title('Mean Return (%)')
        plt.ylabel('Return (%)')
        plt.grid(True, alpha=0.3)
        
        # Biểu đồ tỷ lệ thắng
        plt.subplot(2, 2, 3)
        win_rates = [model_results[name]["win_rate"] * 100 for name in model_names]
        
        plt.bar(range(len(model_names)), win_rates, color=colors[:len(model_names)])
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.title('Win Rate (%)')
        plt.ylabel('Win Rate (%)')
        plt.grid(True, alpha=0.3)
        
        # Biểu đồ tổng số giao dịch
        plt.subplot(2, 2, 4)
        total_trades = [model_results[name]["total_trades"] for name in model_names]
        
        plt.bar(range(len(model_names)), total_trades, color=colors[:len(model_names)])
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.title('Total Trades')
        plt.ylabel('Number of Trades')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        chart_path = output_dir / f"model_comparison_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path)
        logger.info(f"Đã lưu biểu đồ so sánh tại: {chart_path}")
        
        # Tạo bảng so sánh
        comparison_table = []
        for name in model_names:
            result = model_results[name]
            comparison_table.append({
                "Model": name,
                "Mean Reward": f"{result['mean_reward']:.2f} ± {result['std_reward']:.2f}",
                "Mean Return (%)": f"{result['mean_return']:.2f} ± {result['std_return']:.2f}",
                "Win Rate (%)": f"{result['win_rate']*100:.2f}",
                "Total Trades": result['total_trades'],
                "Max Drawdown (%)": f"{result['max_drawdown']*100:.2f}",
                "Avg PnL": f"{result['avg_pnl']:.2f}"
            })
        
        # Lưu bảng so sánh dạng Markdown
        table_path = output_dir / f"model_comparison_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(table_path, 'w', encoding='utf-8') as f:
            # Viết tiêu đề
            f.write("# So sánh các mô hình\n\n")
            f.write(f"Dữ liệu đánh giá: {data_path}\n")
            f.write(f"Số lượng episodes: {episodes}\n\n")
            
            # Viết bảng
            headers = list(comparison_table[0].keys())
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---" for _ in headers]) + " |\n")
            
            for row in comparison_table:
                f.write("| " + " | ".join([str(row[h]) for h in headers]) + " |\n")
        
        logger.info(f"Đã lưu bảng so sánh dạng Markdown tại: {table_path}")
        
        # Hiển thị thông tin tóm tắt
        click.echo("\n" + "="*80)
        click.echo(f"Tóm tắt so sánh {len(model_names)} mô hình:")
        
        # Hiển thị bảng so sánh
        click.echo("\n" + "-"*80)
        header_format = "{:<20} {:<20} {:<20} {:<15} {:<15} {:<15}"
        click.echo(header_format.format("Model", "Mean Reward", "Mean Return (%)", "Win Rate (%)", "Total Trades", "Max Drawdown (%)"))
        click.echo("-"*80)
        
        row_format = "{:<20} {:<20} {:<20} {:<15} {:<15} {:<15}"
        for name in model_names:
            result = model_results[name]
            click.echo(row_format.format(
                name,
                f"{result['mean_reward']:.2f} ± {result['std_reward']:.2f}",
                f"{result['mean_return']:.2f} ± {result['std_return']:.2f}",
                f"{result['win_rate']*100:.2f}",
                result['total_trades'],
                f"{result['max_drawdown']*100:.2f}"
            ))
        
        click.echo("-"*80)
        
        # Mô hình tốt nhất theo các chỉ số khác nhau
        best_reward_model = max(model_names, key=lambda x: model_results[x]['mean_reward'])
        best_return_model = max(model_names, key=lambda x: model_results[x]['mean_return'])
        best_winrate_model = max(model_names, key=lambda x: model_results[x]['win_rate'])
        
        click.echo(f"\nMô hình tốt nhất theo Reward: {best_reward_model}")
        click.echo(f"Mô hình tốt nhất theo Return: {best_return_model}")
        click.echo(f"Mô hình tốt nhất theo Win Rate: {best_winrate_model}")
        
        click.echo(f"\nKết quả so sánh chi tiết: {results_file}")
        click.echo(f"Biểu đồ so sánh: {chart_path}")
        click.echo(f"Bảng so sánh: {table_path}")
        click.echo("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Lỗi khi so sánh mô hình: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Nếu chạy trực tiếp file này
    train_commands()