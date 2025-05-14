import pandas as pd
df = pd.read_parquet("bnb_usdt_20250514_225653.parquet")
df.to_csv("bnb_merged_193704.csv", index=False)
