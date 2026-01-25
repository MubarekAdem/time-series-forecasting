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

warnings.filterwarnings('ignore')

def run_forecasting():
    try:
        print("Loading data...")
        file_path = 'data/processed/TSLA_processed.csv'
        if not os.path.exists(file_path):
            print("File not found.")
            return

        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        # Strict cleaning
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # Ensure numeric
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(inplace=True)
        
        train_data = df[df.index < '2025-01-01']
        test_data = df[df.index >= '2025-01-01']
        
        print(f"Train size: {len(train_data)}")
        print(f"Test size: {len(test_data)}")
        
        if len(test_data) == 0:
            print("Error: Test data is empty!")
            return

        # ARIMA
        print("\n--- Running ARIMA ---")
        forecast_arima = None
        try:
            model_auto = auto_arima(train_data['Close'], seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
            print(f"Best ARIMA order: {model_auto.order}")
            # Use auto_arima model for prediction directly
            forecast_arima = model_auto.predict(n_periods=len(test_data))
            # forecast_arima might be numpy array or series
            if isinstance(forecast_arima, pd.Series):
                forecast_arima = forecast_arima.values
            
            forecast_arima = pd.Series(forecast_arima, index=test_data.index)
        except Exception as e:
            print(f"Auto ARIMA failed: {e}. Trying Statsmodels ARIMA fallback...")
            try:
                model_arima = ARIMA(train_data['Close'], order=(5, 1, 0))
                fit_arima = model_arima.fit()
                forecast_arima = fit_arima.forecast(steps=len(test_data))
                forecast_arima = pd.Series(forecast_arima, index=test_data.index)
            except Exception as e2:
                print(f"Statsmodels ARIMA failed: {e2}")
                # Create dummy forecast to allow script to continue
                forecast_arima = pd.Series([train_data['Close'].mean()] * len(test_data), index=test_data.index)
        
        # LSTM
        print("\n--- Running LSTM ---")
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train_data[['Close']])
        scaled_test = scaler.transform(test_data[['Close']])
        
        seq_length = 60
        
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)
            
        X_train, y_train = create_sequences(scaled_train, seq_length)
        # Prepare test input: last 60 days of train + test data
        inputs = np.concatenate((scaled_train[-seq_length:], scaled_test))
        X_test, y_test = create_sequences(inputs, seq_length)
        
        model_lstm = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        model_lstm.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)
        
        print(f"X_test shape: {X_test.shape}")
        if len(X_test) == 0:
             print("Warning: X_test empty")
             forecast_lstm = pd.Series([0]*len(test_data), index=test_data.index)
        else:
            predictions = model_lstm.predict(X_test)
            predictions = scaler.inverse_transform(predictions)
            forecast_lstm = pd.Series(predictions.flatten(), index=test_data.index)
        
        # Evaluation
        def evaluate(y_true, y_pred, model_name):
            try:
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
            except Exception as e:
                print(f"Evaluation failed for {model_name}: {e}")
                with open(f'{model_name}_eval_error.log', 'w') as f:
                    f.write(str(e))
                return {'Model': model_name, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

        # Ensure indices align
        common_index = test_data.index.intersection(forecast_arima.index).intersection(forecast_lstm.index)
        print(f"Common index size: {len(common_index)}")
        
        res_arima = evaluate(test_data.loc[common_index, 'Close'], forecast_arima.loc[common_index], 'ARIMA')
        res_lstm = evaluate(test_data.loc[common_index, 'Close'], forecast_lstm.loc[common_index], 'LSTM')
        
        results = pd.DataFrame([res_arima, res_lstm])
        print("\n--- Results ---")
        print(results)
        results.to_csv('model_comparison.csv', index=False)
        print("Results saved to model_comparison.csv")

    except Exception as e:
        with open('error.log', 'w') as f:
            f.write(str(e))
            f.write(traceback.format_exc())
        traceback.print_exc()

if __name__ == "__main__":
    run_forecasting()
