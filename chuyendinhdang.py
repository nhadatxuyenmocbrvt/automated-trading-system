import pandas as pd
df = pd.read_parquet("btc_usdt_20250513_115618.parquet")
df.to_csv("btc_sentiment_193704.csv", index=False)
