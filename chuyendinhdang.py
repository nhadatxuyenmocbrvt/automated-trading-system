import pandas as pd
df = pd.read_parquet("btc_usdt_1h.parquet")
df.to_csv("btc_usdt_1h.csv", index=False)
