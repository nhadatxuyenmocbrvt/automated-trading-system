import pandas as pd
df = pd.read_parquet("eth_usdt_20250503_224123.parquet")
df.to_csv("eth_usdt_20250503_212234.csv", index=False)
