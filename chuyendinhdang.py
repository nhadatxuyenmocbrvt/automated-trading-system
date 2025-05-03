import pandas as pd
df = pd.read_parquet("btc_usdt_20250503_193723.parquet")
df.to_csv("btc_usdt_20250501_221115.csv", index=False)
