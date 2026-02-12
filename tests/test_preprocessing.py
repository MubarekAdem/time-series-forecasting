import pytest
import numpy as np
import pandas as pd
from scripts.generate_forecasts import load_data, train_lstm_model

def test_load_data_cleaning():
    # Simulate dirty data
    df = pd.DataFrame({
        'Close': [100, np.inf, -np.inf, np.nan, 105]
    }, index=pd.date_range('2025-01-01', periods=5))
    df.to_csv('test_dirty.csv')
    # Patch load_data to use test_dirty.csv
    def patched_load_data():
        df = pd.read_csv('test_dirty.csv', index_col=0, parse_dates=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(inplace=True)
        return df
    cleaned = patched_load_data()
    assert cleaned.isnull().sum().sum() == 0
    assert cleaned['Close'].min() >= 100
    assert cleaned['Close'].max() <= 105


def test_train_lstm_model_scaling():
    # Simulate data
    df = pd.DataFrame({'Close': np.arange(100, 160)}, index=pd.date_range('2025-01-01', periods=60))
    model, scaler, scaled_data, seq_length = train_lstm_model(df, seq_length=10)
    # Check scaling
    assert np.all(scaled_data >= 0) and np.all(scaled_data <= 1)
    assert scaled_data.shape[0] == 60
    assert scaled_data.shape[1] == 1
    assert seq_length == 10
