import pandas as pd
df = pd.read_parquet("btc_usdt_20250504_203002.parquet")
df.to_csv("btc_usdt_20250504_125533.csv", index=False)
