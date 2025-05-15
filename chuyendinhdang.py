import pandas as pd
df = pd.read_parquet("Dataodactrung.parquet")
df.to_csv("Dataodactrung.csv", index=False)
