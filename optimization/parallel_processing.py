import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import matplotlib.pyplot as plt

# Thêm thư mục gốc vào path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data_processors.feature_engineering.technical_indicators.trend_indicators import TrendIndicators
from data_processors.feature_engineering.technical_indicators.momentum_indicators import MomentumIndicators
from data_processors.feature_engineering.technical_indicators.volume_indicators import VolumeIndicators

def process_chunk(chunk):
    """Xử lý một phần của dữ liệu"""
    # Tạo indicators
    trend = TrendIndicators()
    momentum = MomentumIndicators()
    volume = VolumeIndicators()
    
    # Áp dụng indicators
    result = chunk.copy()
    result = trend.add_indicators(result)
    result = momentum.add_indicators(result)
    result = volume.add_indicators(result)
    
    return result

def process_sequential(data):
    """Xử lý dữ liệu theo cách tuần tự"""
    start_time = time.time()
    
    result = process_chunk(data)
    
    end_time = time.time()
    return result, end_time - start_time

def process_parallel_chunks(data, n_chunks=4):
    """Xử lý dữ liệu song song bằng cách chia nhỏ"""
    start_time = time.time()
    
    # Chia dữ liệu thành các phần
    chunk_size = len(data) // n_chunks
    chunks = [data.iloc[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Xử lý song song
    with ProcessPoolExecutor(max_workers=n_chunks) as executor:
        results = list(executor.map(process_chunk, chunks))
    
    # Kết hợp kết quả
    combined_result = pd.concat(results)
    
    end_time = time.time()
    return combined_result, end_time - start_time

def process_parallel_symbols(symbols_data):
    """Xử lý song song cho nhiều cặp tiền"""
    start_time = time.time()
    
    # Xử lý song song
    with ThreadPoolExecutor(max_workers=len(symbols_data)) as executor:
        futures = {symbol: executor.submit(process_chunk, data) 
                  for symbol, data in symbols_data.items()}
        
        results = {symbol: future.result() for symbol, future in futures.items()}
    
    end_time = time.time()
    return results, end_time - start_time

def compare_methods():
    """So sánh hiệu suất các phương pháp xử lý"""
    # Tạo dữ liệu lớn cho kiểm thử
    print("Đang tạo dữ liệu kiểm thử...")
    dates = pd.date_range('2020-01-01', periods=10000, freq='H')
    data = pd.DataFrame({
        'open': np.random.normal(40000, 1000, 10000),
        'high': np.random.normal(41000, 1000, 10000),
        'low': np.random.normal(39000, 1000, 10000),
        'close': np.random.normal(40500, 1000, 10000),
        'volume': np.random.normal(1000, 200, 10000)
    }, index=dates)
    
    # Tạo nhiều cặp tiền
    symbols_data = {
        'BTC/USDT': data.copy(),
        'ETH/USDT': data.copy() * 0.1,
        'XRP/USDT': data.copy() * 0.01,
        'ADA/USDT': data.copy() * 0.001
    }
    
    # Đo lường thời gian xử lý tuần tự
    print("Đang xử lý tuần tự...")
    _, sequential_time = process_sequential(data)
    print(f"Thời gian xử lý tuần tự: {sequential_time:.2f} giây")
    
    # Đo lường thời gian xử lý song song bằng cách chia nhỏ
    print("Đang xử lý song song bằng cách chia nhỏ...")
    _, parallel_chunks_time = process_parallel_chunks(data, n_chunks=multiprocessing.cpu_count())
    print(f"Thời gian xử lý song song (chunks): {parallel_chunks_time:.2f} giây")
    
    # Đo lường thời gian xử lý song song cho nhiều cặp tiền
    print("Đang xử lý song song cho nhiều cặp tiền...")
    _, parallel_symbols_time = process_parallel_symbols(symbols_data)
    print(f"Thời gian xử lý song song (symbols): {parallel_symbols_time:.2f} giây")
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(10, 6))
    plt.bar(['Sequential', 'Parallel (Chunks)', 'Parallel (Symbols)'], 
            [sequential_time, parallel_chunks_time, parallel_symbols_time])
    plt.ylabel('Processing Time (seconds)')
    plt.title('Processing Time Comparison')
    plt.savefig('optimization/parallel_processing_comparison.png')
    
    # Tính toán speedup
    print("\nSpeedup factors:")
    print(f"Parallel (Chunks) speedup: {sequential_time / parallel_chunks_time:.2f}x")
    print(f"Parallel (Symbols) speedup: {sequential_time / parallel_symbols_time:.2f}x")
    
    return {
        'sequential_time': sequential_time,
        'parallel_chunks_time': parallel_chunks_time,
        'parallel_symbols_time': parallel_symbols_time
    }

if __name__ == "__main__":
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs("optimization", exist_ok=True)
    
    # So sánh phương pháp xử lý
    print("Đang so sánh các phương pháp xử lý...")
    compare_methods()