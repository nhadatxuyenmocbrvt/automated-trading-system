import pandas as pd
df = pd.read_parquet("eth_usdt_20250508_170923.parquet")
df.to_csv("eth4m_output.csv", index=False)
