"""
Dữ liệu mẫu cho các unit test của module technical_indicators.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Lấy logger từ __init__.py
from tests.technical_indicators import logger

def generate_sample_price_data(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Tạo dữ liệu giá mẫu cho việc thử nghiệm.
    
    Args:
        n_samples: Số lượng mẫu
        seed: Seed ngẫu nhiên
        
    Returns:
        DataFrame chứa dữ liệu giá OHLCV
    """
    logger.info(f"Tạo dữ liệu giá mẫu với {n_samples} mẫu, seed={seed}")
    np.random.seed(seed)
    
    # Tạo dữ liệu ngày (datetime)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Tạo giá close với xu hướng tăng nhẹ + nhiễu ngẫu nhiên
    close = np.cumsum(np.random.normal(0.001, 0.02, n_samples)) + 100
    
    # Tạo giá cao và thấp dựa trên close
    high = close + np.random.uniform(0.01, 0.05, n_samples) * close
    low = close - np.random.uniform(0.01, 0.05, n_samples) * close
    
    # Tạo giá mở nằm giữa high và low
    open_price = low + np.random.uniform(0, 1, n_samples) * (high - low)
    
    # Tạo khối lượng với sự biến động ngẫu nhiên
    volume = np.random.normal(1000000, 500000, n_samples)
    volume = np.abs(volume)  # Đảm bảo khối lượng không âm
    
    # Tạo DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Đặt timestamp làm index
    df.set_index('timestamp', inplace=True)
    
    logger.debug(f"Đã tạo DataFrame với shape={df.shape}")
    return df

def generate_trending_price_data(n_samples: int = 100, trend_strength: float = 0.01, seed: int = 42) -> pd.DataFrame:
    """
    Tạo dữ liệu giá mẫu có xu hướng rõ ràng.
    
    Args:
        n_samples: Số lượng mẫu
        trend_strength: Độ mạnh xu hướng (dương cho xu hướng tăng, âm cho xu hướng giảm)
        seed: Seed ngẫu nhiên
        
    Returns:
        DataFrame chứa dữ liệu giá OHLCV
    """
    logger.info(f"Tạo dữ liệu trending với {n_samples} mẫu, trend_strength={trend_strength}, seed={seed}")
    np.random.seed(seed)
    
    # Tạo dữ liệu ngày (datetime)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Tạo giá close với xu hướng rõ ràng + nhiễu ngẫu nhiên
    trend = np.arange(n_samples) * trend_strength  # Xu hướng tuyến tính
    noise = np.random.normal(0, 0.01, n_samples)  # Nhiễu thấp hơn để xu hướng rõ ràng
    close = 100 + trend + noise
    
    # Tạo giá cao và thấp dựa trên close
    high = close + np.random.uniform(0.005, 0.02, n_samples) * close
    low = close - np.random.uniform(0.005, 0.02, n_samples) * close
    
    # Tạo giá mở nằm giữa high và low
    open_price = low + np.random.uniform(0, 1, n_samples) * (high - low)
    
    # Tạo khối lượng tăng theo giá (cho xu hướng tăng) hoặc giảm theo giá (cho xu hướng giảm)
    if trend_strength > 0:
        volume_trend = np.linspace(800000, 1200000, n_samples)
    else:
        volume_trend = np.linspace(1200000, 800000, n_samples)
    
    volume = volume_trend + np.random.normal(0, 100000, n_samples)
    volume = np.abs(volume)  # Đảm bảo khối lượng không âm
    
    # Tạo DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Đặt timestamp làm index
    df.set_index('timestamp', inplace=True)
    
    logger.debug(f"Đã tạo DataFrame trending với shape={df.shape}")
    return df

def generate_volatile_price_data(n_samples: int = 100, volatility: float = 0.03, seed: int = 42) -> pd.DataFrame:
    """
    Tạo dữ liệu giá mẫu có biến động cao.
    
    Args:
        n_samples: Số lượng mẫu
        volatility: Mức độ biến động
        seed: Seed ngẫu nhiên
        
    Returns:
        DataFrame chứa dữ liệu giá OHLCV
    """
    logger.info(f"Tạo dữ liệu volatile với {n_samples} mẫu, volatility={volatility}, seed={seed}")
    np.random.seed(seed)
    
    # Tạo dữ liệu ngày (datetime)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Tạo giá close với biến động cao
    close = np.cumsum(np.random.normal(0, volatility, n_samples)) + 100
    
    # Tạo giá cao và thấp với khoảng cách rộng hơn so với close
    high = close + np.random.uniform(0.02, 0.08, n_samples) * close
    low = close - np.random.uniform(0.02, 0.08, n_samples) * close
    
    # Tạo giá mở nằm giữa high và low
    open_price = low + np.random.uniform(0, 1, n_samples) * (high - low)
    
    # Tạo khối lượng với sự biến động ngẫu nhiên cao
    volume = np.random.normal(1000000, 800000, n_samples)
    volume = np.abs(volume)  # Đảm bảo khối lượng không âm
    
    # Tạo DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Đặt timestamp làm index
    df.set_index('timestamp', inplace=True)
    
    logger.debug(f"Đã tạo DataFrame volatile với shape={df.shape}")
    return df