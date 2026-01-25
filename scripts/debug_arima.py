import pandas as pd
import numpy as np
from pmdarima import auto_arima
import os

def debug_arima():
    file_path = 'data/processed/TSLA_processed.csv'
    if not os.path.exists(file_path):
        print("File not found")
        return

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print("Original shape:", df.shape)
    print("Original NaNs:\n", df.isnull().sum())

    df.dropna(inplace=True)
    print("Shape after dropna:", df.shape)
    print("NaNs after dropna:\n", df.isnull().sum())

    train_data = df[df.index < '2025-01-01']
    print("Train shape:", train_data.shape)
    
    # Check for Inf
    print("Inf in Close:", np.isinf(train_data['Close']).sum())

    try:
        print("Starting auto_arima...")
        model = auto_arima(train_data['Close'], seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
        print(model.summary())
    except Exception as e:
        print("ARIMA failed:", e)

if __name__ == "__main__":
    debug_arima()
