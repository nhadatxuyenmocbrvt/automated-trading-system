import pandas as pd
df = pd.read_parquet("btc_sentiment_20250511_193704.parquet")
df.to_csv("btc_sentiment_193704.csv", index=False)
