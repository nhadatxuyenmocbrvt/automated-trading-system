import pandas as pd
df = pd.read_parquet("btc_usdt_20250509_132732.parquet")
df.to_csv("btc3y_output.csv", index=False)
