import pandas as pd
df = pd.read_csv('data/processed/TSLA_processed.csv', index_col=0, parse_dates=True)
print("Start:", df.index.min())
print("End:", df.index.max())
print("Shape:", df.shape)
test = df[df.index >= '2025-01-01']
print("Test shape:", test.shape)
print("Test NaNs:\n", test.isnull().sum())
