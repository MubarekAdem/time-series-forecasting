"""
TSLA Stock Price Forecasting Model Comparison

This script calculates and compares performance metrics for ARIMA and LSTM models
on historical TSLA stock data. Refactored for modularity and reproducibility.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import warnings
import traceback
import random
import tensorflow as tf

# Set random seeds for reproducibility
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

warnings.filterwarnings('ignore')

def load_and_clean_data(file_path):
    """
    Load and perform strict cleaning on historical price data.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Cleaned price data with DatetimeIndex.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Handle infinities and missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Ensure 'Close' column is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)
    
    return df

def train_arima_model(train_data):
    """
    Fit an ARIMA model automatically using pmdarima.
    
    Args:
        train_data (pd.Series): Training price data.
        
    Returns:
        pmdarima.arima.ARIMA: Fitted ARIMA model.
    """
    print("\nTraining ARIMA model...")
    model = auto_arima(
        train_data, 
        seasonal=False, 
        trace=False, 
        error_action='ignore', 
        suppress_warnings=True,
        stepwise=True
    )
    print(f"Optimal ARIMA order: {model.order}")
    return model

def create_sequences(data, seq_length):
    """
    Create input sequences for LSTM training/testing.
    
    Args:
        data (np.ndarray): Scaled price data.
        seq_length (int): Length of each input sequence.
        
    Returns:
        tuple: (X, y) as numpy arrays.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_lstm_model(train_series, seq_length=60):
    """
    Scale data and train an LSTM model.
    
    Args:
        train_series (pd.Series): Training price data.
        seq_length (int): Lookback period.
        
    Returns:
        tuple: (trained_model, scaler, scaled_train_data)
    """
    print("\nTraining LSTM model...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(train_series.values.reshape(-1, 1))
    
    X, y = create_sequences(scaled_data, seq_length)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=5, verbose=0)
    
    return model, scaler, scaled_data

def evaluate_model(y_true, y_pred, model_name):
    """
    Calculate error metrics for model predictions.
    
    Args:
        y_true (pd.Series): Actual prices.
        y_pred (pd.Series): Predicted prices.
        model_name (str): Name of the model for identification.
        
    Returns:
        dict: Performance metrics (MAE, RMSE, MAPE).
    """
    try:
        # Align indices
        common_idx = y_true.index.intersection(y_pred.index)
        y_t = y_true.loc[common_idx]
        y_p = y_pred.loc[common_idx]
        
        mae = mean_absolute_error(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mape = np.mean(np.abs((y_t - y_p) / y_t)) * 100
        
        return {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return {'Model': model_name, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

def main():
    """Main execution flow for TSLA stock price forecasting comparison."""
    try:
        print("="*60)
        print("TSLA Model Comparison: ARIMA vs LSTM")
        print("="*60)
        
        # 1. Load data
        file_path = 'data/processed/TSLA_processed.csv'
        df = load_and_clean_data(file_path)
        
        # 2. Split data (Chronological)
        split_date = '2025-01-01'
        train_df = df[df.index < split_date]
        test_df = df[df.index >= split_date]
        
        print(f"Training period: {train_df.index.min().date()} to {train_df.index.max().date()}")
        print(f"Testing period:  {test_df.index.min().date()} to {test_df.index.max().date()}")
        
        if len(test_df) == 0:
            print("Error: Testing period (2025+) contains no data.")
            return

        # 3. ARIMA Forecast
        arima_model = train_arima_model(train_df['Close'])
        arima_preds = arima_model.predict(n_periods=len(test_df))
        forecast_arima = pd.Series(arima_preds, index=test_df.index)
        
        # 4. LSTM Forecast
        seq_length = 60
        lstm_model, scaler, scaled_train = train_lstm_model(train_df['Close'], seq_length)
        
        # Prepare test sequences
        scaled_test = scaler.transform(test_df[['Close']])
        all_inputs = np.concatenate((scaled_train[-seq_length:], scaled_test))
        X_test, _ = create_sequences(all_inputs, seq_length)
        
        lstm_preds = lstm_model.predict(X_test, verbose=0)
        lstm_preds = scaler.inverse_transform(lstm_preds)
        forecast_lstm = pd.Series(lstm_preds.flatten(), index=test_df.index)
        
        # 5. Evaluation
        results = []
        results.append(evaluate_model(test_df['Close'], forecast_arima, 'ARIMA'))
        results.append(evaluate_model(test_df['Close'], forecast_lstm, 'LSTM'))
        
        results_df = pd.DataFrame(results)
        print("\nModel Comparison Results:")
        print(results_df)
        
        # 6. Save results
        results_df.to_csv('model_comparison.csv', index=False)
        print("\nResults archived to model_comparison.csv")
        
    except Exception as e:
        print(f"Execution failed: {e}")
        with open('forecasting_error.log', 'w') as f:
            f.write(traceback.format_exc())
        traceback.print_exc()

if __name__ == "__main__":
    main()
