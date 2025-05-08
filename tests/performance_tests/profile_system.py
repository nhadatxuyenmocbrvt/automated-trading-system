import cProfile
import pstats
import io
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Thêm thư mục gốc vào path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.system_config import SystemConfig
from data_collectors.exchange_api.binance_connector import BinanceConnector
from data_processors.data_pipeline import DataPipeline
from environments.trading_gym.trading_env import TradingEnv
from models.agents.dqn_agent import DQNAgent
from backtesting.backtester import Backtester

def load_test_data():
    """Tải dữ liệu kiểm thử"""
    # Trong triển khai thực tế, bạn sẽ tải dữ liệu thực
    data_path = os.path.join(project_root, "tests/data/test_data.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        # Tạo dữ liệu giả lập
        dates = pd.date_range('2023-01-01', periods=1000, freq='H')
        data = pd.DataFrame({
            'open': np.random.normal(40000, 1000, 1000),
            'high': np.random.normal(41000, 1000, 1000),
            'low': np.random.normal(39000, 1000, 1000),
            'close': np.random.normal(40500, 1000, 1000),
            'volume': np.random.normal(1000, 200, 1000)
        }, index=dates)
        data.to_csv(data_path)
        return data

def profile_data_processing():
    """Phân tích hiệu suất xử lý dữ liệu"""
    config = SystemConfig()
    data = load_test_data()
    pipeline = DataPipeline(config)
    
    # Đo thời gian
    start_time = time.time()
    
    # Profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Xử lý dữ liệu
    processed_data = pipeline.process(data)
    
    profiler.disable()
    end_time = time.time()
    
    # In kết quả
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print(f"Data processing took {end_time - start_time:.2f} seconds")
    print(s.getvalue())
    
    # Lưu report
    with open("performance_reports/data_processing_profile.txt", "w") as f:
        f.write(s.getvalue())
    
    return processed_data

def profile_agent_training():
    """Phân tích hiệu suất huấn luyện agent"""
    config = SystemConfig()
    data = profile_data_processing()  # Sử dụng dữ liệu đã xử lý
    
    # Tạo môi trường
    env = TradingEnv(
        data=data,
        initial_balance=10000,
        commission=0.001
    )
    
    # Khởi tạo agent
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        config=config
    )
    
    # Đo thời gian
    start_time = time.time()
    
    # Profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Huấn luyện agent
    agent.train(env, episodes=10, batch_size=32)
    
    profiler.disable()
    end_time = time.time()
    
    # In kết quả
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print(f"Agent training took {end_time - start_time:.2f} seconds")
    print(s.getvalue())
    
    # Lưu report
    with open("performance_reports/agent_training_profile.txt", "w") as f:
        f.write(s.getvalue())

def profile_backtesting():
    """Phân tích hiệu suất backtest"""
    config = SystemConfig()
    data = load_test_data()
    
    # Tạo môi trường
    env = TradingEnv(
        data=data,
        initial_balance=10000,
        commission=0.001
    )
    
    # Khởi tạo agent
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        config=config
    )
    
    # Khởi tạo backtester
    backtester = Backtester(env, agent)
    
    # Đo thời gian
    start_time = time.time()
    
    # Profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Chạy backtest
    results = backtester.run()
    
    profiler.disable()
    end_time = time.time()
    
    # In kết quả
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print(f"Backtesting took {end_time - start_time:.2f} seconds")
    print(s.getvalue())
    
    # Lưu report
    with open("performance_reports/backtesting_profile.txt", "w") as f:
        f.write(s.getvalue())

def run_all_profiles():
    """Chạy tất cả các phân tích hiệu suất"""
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs("performance_reports", exist_ok=True)
    
    print("Profiling data processing...")
    profile_data_processing()
    
    print("\nProfileing agent training...")
    profile_agent_training()
    
    print("\nProfileing backtesting...")
    profile_backtesting()
    
    print("\nAll profiling complete. Reports saved to performance_reports/")

if __name__ == "__main__":
    run_all_profiles()