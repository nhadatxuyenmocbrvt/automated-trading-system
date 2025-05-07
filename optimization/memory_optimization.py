import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import psutil
import matplotlib.pyplot as plt

# Thêm thư mục gốc vào path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data_processors.feature_engineering.technical_indicators.trend_indicators import TrendIndicators

def analyze_memory_usage():
    """Phân tích sử dụng bộ nhớ của các hàm chính"""
    # Tạo dữ liệu lớn cho kiểm thử
    print("Tạo dữ liệu kiểm thử...")
    dates = pd.date_range('2020-01-01', periods=10000, freq='H')
    data = pd.DataFrame({
        'open': np.random.normal(40000, 1000, 10000),
        'high': np.random.normal(41000, 1000, 10000),
        'low': np.random.normal(39000, 1000, 10000),
        'close': np.random.normal(40500, 1000, 10000),
        'volume': np.random.normal(1000, 200, 10000)
    }, index=dates)
    
    # Đo lường sử dụng bộ nhớ ban đầu
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"Bộ nhớ ban đầu: {initial_memory:.2f} MB")
    
    # Test indicators
    trend_indicators = TrendIndicators()
    
    # Đo lường bộ nhớ sau khi tạo indicators
    before_indicators = process.memory_info().rss / 1024 / 1024
    print(f"Bộ nhớ trước khi tạo indicators: {before_indicators:.2f} MB")
    
    # Áp dụng indicators
    data_with_features = trend_indicators.add_indicators(data)
    
    # Đo lường bộ nhớ sau khi tạo indicators
    after_indicators = process.memory_info().rss / 1024 / 1024
    print(f"Bộ nhớ sau khi tạo indicators: {after_indicators:.2f} MB")
    print(f"Tăng bộ nhớ: {after_indicators - before_indicators:.2f} MB")
    
    # Vẽ biểu đồ sử dụng bộ nhớ
    plt.figure(figsize=(10, 6))
    plt.bar(['Initial', 'Before Indicators', 'After Indicators'], 
            [initial_memory, before_indicators, after_indicators])
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Analysis')
    plt.savefig('optimization/memory_usage.png')
    
    # Đề xuất tối ưu hóa
    print("\nĐề xuất tối ưu hóa bộ nhớ:")
    print("1. Sử dụng dtype phù hợp cho DataFrame")
    print("2. Sử dụng inplace=True khi có thể")
    print("3. Xóa các cột trung gian không cần thiết")
    print("4. Sử dụng chunking cho dữ liệu lớn")
    
    return {
        'initial_memory': initial_memory,
        'before_indicators': before_indicators,
        'after_indicators': after_indicators,
        'increase': after_indicators - before_indicators
    }

def optimize_dataframe(df):
    """Tối ưu hóa bộ nhớ cho DataFrame"""
    memory_before = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"Bộ nhớ trước khi tối ưu: {memory_before:.2f} MB")
    
    # Tối ưu hóa các cột số
    for col in df.select_dtypes(include=['float']).columns:
        # Chuyển đổi float64 thành float32 nếu phù hợp
        df[col] = df[col].astype('float32')
    
    for col in df.select_dtypes(include=['int']).columns:
        # Xác định loại int nhỏ nhất có thể sử dụng
        c_min = df[col].min()
        c_max = df[col].max()
        
        if c_min >= 0:
            if c_max < 255:
                df[col] = df[col].astype('uint8')
            elif c_max < 65535:
                df[col] = df[col].astype('uint16')
            elif c_max < 4294967295:
                df[col] = df[col].astype('uint32')
        else:
            if c_min > -128 and c_max < 127:
                df[col] = df[col].astype('int8')
            elif c_min > -32768 and c_max < 32767:
                df[col] = df[col].astype('int16')
            elif c_min > -2147483648 and c_max < 2147483647:
                df[col] = df[col].astype('int32')
    
    memory_after = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"Bộ nhớ sau khi tối ưu: {memory_after:.2f} MB")
    print(f"Giảm: {memory_before - memory_after:.2f} MB ({(1 - memory_after/memory_before) * 100:.2f}%)")
    
    return df

if __name__ == "__main__":
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs("optimization", exist_ok=True)
    
    # Phân tích sử dụng bộ nhớ
    print("Đang phân tích sử dụng bộ nhớ...")
    memory_analysis = analyze_memory_usage()
    
    # Tạo dữ liệu kiểm thử
    print("\nĐang kiểm thử tối ưu hóa DataFrame...")
    dates = pd.date_range('2020-01-01', periods=10000, freq='H')
    test_df = pd.DataFrame({
        'open': np.random.normal(40000, 1000, 10000),
        'high': np.random.normal(41000, 1000, 10000),
        'low': np.random.normal(39000, 1000, 10000),
        'close': np.random.normal(40500, 1000, 10000),
        'volume': np.random.normal(1000, 200, 10000).astype('int'),
        'trades': np.random.randint(0, 1000, 10000)
    }, index=dates)
    
    # Tối ưu hóa DataFrame
    optimized_df = optimize_dataframe(test_df)