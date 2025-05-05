import pandas as pd
import numpy as np

# Đường dẫn tuyệt đối đến file .parquet
file_path = r"E:\AI_AGENT\automated-trading-system\data\processed\btc_usdt_20250505_011446.parquet"

# Đọc file parquet
df = pd.read_parquet(file_path)

print("✅ Tổng số dòng:", len(df))
print("✅ Các cột:", df.columns.tolist())

# 1. Kiểm tra missing value
missing = df.isnull().sum()
print("\n🔍 Thiếu dữ liệu:\n", missing[missing > 0])

# 2. Kiểm tra đặc trưng có giá trị cực đoan
extreme = df.select_dtypes(include='number').apply(lambda col: (
    (col > 1e6).sum(), (col < -1e6).sum()))
print("\n⚠️ Cột có giá trị cực lớn hoặc cực nhỏ:\n", extreme)

# 3. Thống kê reward nếu có
reward_cols = [col for col in df.columns if 'reward' in col or col == 'target']
for col in reward_cols:
    print(f"\n📊 Thống kê cho cột: {col}")
    print(df[col].describe())
    print("Top 5 lớn nhất:\n", df[col].nlargest(5))
