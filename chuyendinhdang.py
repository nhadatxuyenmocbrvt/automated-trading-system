import pandas as pd
df = pd.read_parquet("input6m.parquet")
df.to_csv("btc6m_output.csv", index=False)
