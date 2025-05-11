import pandas as pd
df = pd.read_parquet("xrp_usdt_20250510_151624.parquet")
df.to_csv("xrp_output.csv", index=False)
